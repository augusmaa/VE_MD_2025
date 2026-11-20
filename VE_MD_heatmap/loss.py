"""Classification, stage-wise heatmap MSE and frame-level latent MMD."""

import torch
from torch import nn
from torch.nn import functional as F


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


class StageHeatmapLossMSE(nn.Module):
    """Sum the mean squared error of sigmoid heatmaps over all decoder stages."""

    def forward(self, stage_logits, target):
        if target.ndim == 5:
            target = target.flatten(0, 1)
        if not stage_logits or any(prediction.shape != target.shape for prediction in stage_logits):
            raise ValueError("Stage predictions and target heatmaps must have matching shapes.")
        return sum(F.mse_loss(prediction.sigmoid(), target) for prediction in stage_logits)


class VEMDLoss(nn.Module):
    """Classification + weighted body/face heatmap losses + weighted latent MMD."""

    def __init__(self, person_weight=1.0, face_weight=1.0, mmd_weight=1.0, mmd_sigma=1.0):
        super().__init__()
        if min(person_weight, face_weight, mmd_weight) < 0:
            raise ValueError("Loss weights must be nonnegative.")
        self.weights = {"person": person_weight, "face": face_weight}
        self.mmd_weight = mmd_weight
        self.heatmap_loss = StageHeatmapLossMSE()
        self.mmd = MMDLoss(mmd_sigma)

    def forward(self, outputs, batch):
        classification = F.cross_entropy(outputs["preds_emotion"], batch["label_emotion"])
        structural = classification.new_zeros(())
        for branch in ("person", "face"):
            key = f"heatmap_{branch}_logits"
            if key in outputs:
                structural = structural + self.weights[branch] * self.heatmap_loss(outputs[key], batch[f"limb_{branch}"])
        z = outputs["z"]
        regularization = (self.mmd(z, torch.randn_like(z)) * self.mmd_weight
                          if self.mmd_weight else z.sum() * 0.0)
        total = classification + structural + regularization
        return total, {"loss": total, "classification": classification,
                       "structural": structural, "mmd": regularization}


