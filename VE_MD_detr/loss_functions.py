import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

def loss_skeleton_adjacency_with_mask(
    pred_coords: torch.Tensor,   # (B, num_preds, 4*K), in [0,1]
    pred_adj:    torch.Tensor,   # (B, num_preds, N^2), after nn.Sigmoid()
    gt_coords:   torch.Tensor,   # (B, frames, num_gts, 4*K), in pixel coords or -1
    gt_adj:      torch.Tensor,   # (B, frames, num_gts, N^2), in {0,1}
    lambda_l1:   float = 1.0,
    lambda_adj:  float = 1.0,
    image_size:  float = 224.0,   # for normalizing GT coords
    mask_value:  float = -1.0     # the value in gt_coords that means “missing”
):
    """
    Returns:
      total_loss: scalar tensor
      matches:   list of dicts with 'pred_indices' and 'gt_indices' per batch
    """
    B, num_preds, coord_dim = pred_coords.shape
    _, _, adj_dim    = pred_adj.shape

    # flatten out frames → batch'
    bs, frames, num_gts, _ = gt_coords.shape
    gt_coords = gt_coords.view(bs*frames, num_gts, coord_dim)
    gt_adj    = gt_adj   .view(bs*frames, num_gts, adj_dim)

    all_losses = []
    all_matches = []

    for b in range(B):
        p_c = pred_coords[b]   # (num_preds, coord_dim)
        p_a = pred_adj[b]      # (num_preds, adj_dim)
        g_c = gt_coords[b]     # (num_gts,   coord_dim)
        g_a = gt_adj[b]        # (num_gts,   adj_dim)

        if g_c.size(0) == 0 or num_preds == 0:
            all_matches.append({'pred_indices': [], 'gt_indices': []})
            all_losses.append(p_c.new_tensor(0.0))
            continue

        # build cost matrix of size (num_preds x num_gts)
        cost = torch.zeros((num_preds, g_c.size(0)), device=p_c.device)

        for i in range(num_preds):
            pi = p_c[i]        # ( coord_dim,)
            ai = p_a[i]        # ( adj_dim,)

            for j in range(g_c.size(0)):
                gj = g_c[j]    # ( coord_dim,)
                aj = g_a[j]    # ( adj_dim,)

                # —— coordinate cost —— 
                # mask out missing GT coords
                valid = (gj != mask_value).float()      # 1 for valid coords
                num_valid = valid.sum().clamp(min=1.0)

                # normalize GT into [0,1]
                gj_norm = gj / image_size

                # per-element SmoothL1
                per_elem = F.smooth_l1_loss(pi, gj_norm, reduction='none')
                coord_cost = (per_elem * valid).sum() / num_valid

                # —— adjacency cost —— 
                # use standard BCE sigmoid‐output
                adj_cost = F.binary_cross_entropy(ai, aj, reduction='mean')

                cost[i, j] = lambda_l1 * coord_cost + lambda_adj * adj_cost

        # Hungarian matching
        cost_np = cost.detach().cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_np)

        # final loss is just the average cost over matched pairs
        matched_loss = cost[row_ind, col_ind].mean()
        all_matches.append({
            'pred_indices': row_ind.tolist(),
            'gt_indices':   col_ind.tolist()
        })
        all_losses.append(matched_loss)

    total_loss = torch.stack(all_losses).mean()
    return total_loss, all_matches




## MMD Loss 
class MMDLoss(nn.Module):
    def __init__(self, kernel='rbf', sigma=1.0):
        super(MMDLoss, self).__init__()
        self.kernel = kernel
        self.sigma = sigma

    def gaussian_kernel(self, x, y):
        beta = 1.0 / (2.0 * self.sigma ** 2)
        dist = torch.cdist(x, y, p=2).pow(2)
        return torch.exp(-beta * dist)

    def linear_kernel(self, x, y):
        return torch.mm(x, y.t())

    def forward(self, x, y):
        if self.kernel == 'rbf':
            Kxx = self.gaussian_kernel(x, x)
            Kyy = self.gaussian_kernel(y, y)
            Kxy = self.gaussian_kernel(x, y)
        elif self.kernel == 'linear':
            Kxx = self.linear_kernel(x, x)
            Kyy = self.linear_kernel(y, y)
            Kxy = self.linear_kernel(x, y)
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}")

        mmd = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
        return mmd

