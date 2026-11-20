"""Hungarian structural supervision and frame-level latent MMD."""

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


def loss_skeleton_adjacency_with_mask(pred_coords, 
                                      pred_adj, 
                                      gt_coords, 
                                      gt_adj,
                                      lambda_l1=1.0, 
                                      lambda_adj=0.5,
                                      image_size=224.0, 
                                      mask_value=-1.0):
    """Match predicted sets to valid people; skip completely missing/padded sets.

    Predictions: (B*T,Q,4*K), (B*T,Q,K*K).
    Targets: (B,T,N,4*K), (B,T,N,K,K), coordinates in resized-image pixels.
    All frames, including frames without annotations, contribute to the mean.
    """
    batch, frames, people, coord_dim = gt_coords.shape
    if pred_coords.size(0) != batch * frames or pred_coords.size(2) != coord_dim:
        raise ValueError("Prediction and target coordinate dimensions are inconsistent.")
    targets = gt_coords.reshape(batch * frames, people, coord_dim)
    adjacency = gt_adj.reshape(batch * frames, people, pred_adj.size(-1))
    losses, matches = [], []
    for pc, pa, gc, ga in zip(pred_coords, pred_adj, targets, adjacency):
        valid_people = (gc != mask_value).any(dim=-1)
        original_indices = valid_people.nonzero(as_tuple=False).flatten()
        gc, ga = gc[valid_people], ga[valid_people]
        if len(gc) == 0:
            # Keep both prediction heads in the autograd graph for DDP.
            losses.append((pc.sum() + pa.sum()) * 0.0)
            matches.append({"pred_indices": [], "gt_indices": []})
            continue
        with torch.no_grad():
            # Compute one target column at a time; avoid a Q*N*K*K allocation.
            columns = []
            for coords, adj in zip(gc, ga):
                valid = coords != mask_value
                coordinate_cost = F.smooth_l1_loss(
                    pc, (coords / image_size).expand_as(pc), reduction="none"
                )
                coordinate_cost = (coordinate_cost * valid).sum(-1) / valid.sum().clamp_min(1)
                adjacency_cost = F.binary_cross_entropy(pa, adj.expand_as(pa), reduction="none").mean(-1)
                columns.append(lambda_l1 * coordinate_cost + lambda_adj * adjacency_cost)
            cost = torch.stack(columns, dim=1)
            rows, cols = linear_sum_assignment(cost.cpu().numpy())
        rows = torch.tensor(rows, dtype=torch.long, device=pc.device)
        cols = torch.tensor(cols, dtype=torch.long, device=pc.device)
        valid = gc[cols] != mask_value
        coordinate_loss = F.smooth_l1_loss(pc[rows], gc[cols] / image_size, reduction="none")
        coordinate_loss = (coordinate_loss * valid).sum(-1) / valid.sum(-1).clamp_min(1)
        adjacency_loss = F.binary_cross_entropy(pa[rows], ga[cols], reduction="none").mean(-1)
        losses.append((lambda_l1 * coordinate_loss + lambda_adj * adjacency_loss).mean())
        matches.append({"pred_indices": rows.tolist(), "gt_indices": original_indices[cols].tolist()})
    return torch.stack(losses).mean(), matches


class MMDLoss(nn.Module):
    """Biased RBF MMD, treating each flattened frame latent as one sample."""

    def __init__(self, sigma=1.0):
        super().__init__()
        if sigma <= 0:
            raise ValueError("MMD sigma must be positive.")
        self.sigma = sigma

    def gaussian_kernel(self, x, y):
        # Matrix products support CPU, CUDA, and MPS without torch.cdist.
        distances = (x.square().sum(1, keepdim=True) + y.square().sum(1)[None, :] - 2 * x @ y.T).clamp_min(0)
        return torch.exp(-distances / (2 * self.sigma**2))

    def forward(self, x, y):
        x, y = x.flatten(1), y.flatten(1)
        return (self.gaussian_kernel(x, x).mean() + self.gaussian_kernel(y, y).mean()
                - 2 * self.gaussian_kernel(x, y).mean())


class VEMDLoss(nn.Module):
    """Classification + weighted structural supervision + weighted latent MMD."""

    def __init__(self, 
                 image_size=224, 
                 pose_weight=0.1, 
                 mmd_weight=1.0, 
                 mmd_sigma=1.0):
        super().__init__()
        if pose_weight < 0 or mmd_weight < 0:
            raise ValueError("Loss weights must be nonnegative.")
        self.image_size = image_size
        self.pose_weight = pose_weight
        self.mmd_weight = mmd_weight
        self.mmd = MMDLoss(mmd_sigma)

    def forward(self, outputs, batch):
        classification = F.cross_entropy(outputs["preds_emotion"], batch["label_emotion"])
        structural = classification.new_zeros(())
        for branch in ("person", "face"):
            if f"skeleton_{branch}_pred" in outputs:
                loss, _ = loss_skeleton_adjacency_with_mask(
                    outputs[f"skeleton_{branch}_pred"], outputs[f"adjacency_{branch}_pred"],
                    batch[f"skeleton_{branch}"], batch[f"adjacency_{branch}"],
                    image_size=self.image_size,
                )
                structural = structural + self.pose_weight * loss
        z = outputs["z"]
        regularization = (self.mmd(z, torch.randn_like(z)) * self.mmd_weight
                          if self.mmd_weight else z.sum() * 0.0)
        total = classification + structural + regularization
        return total, {"loss": total, "classification": classification,
                       "structural": structural, "mmd": regularization}
