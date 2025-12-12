import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapLossMSE(nn.Module):
    def __init__(self, reduction='mean'):
        super(HeatmapLossMSE, self).__init__()
        self.reduction = reduction

    def mse_loss(self, pred, gt):
        """
        Compute MSE loss between prediction and ground truth.
        
        Args:
            pred (tensor): Predicted heatmap. Shape: (B, C, H, W).
            gt (tensor): Ground truth heatmap. Shape: (B, C, H, W).
        
        Returns:
            loss (tensor): MSE loss computed with a sum reduction.
        """
        return F.mse_loss(pred, gt, reduction=self.reduction)

    def forward(self, heatmap_preds, heatmap_gt):
        
        # Handle batched input with multiple frames if needed
        if len(heatmap_gt.size()) > 4:
            bs, fr, c, h, w = heatmap_gt.size()
            heatmap_gt = heatmap_gt.view(-1, c, h, w)

        heatmap_loss = self.mse_loss(heatmap_preds, heatmap_gt)
        return heatmap_loss
    


class HeatmapLossMSEmask(nn.Module):
    def __init__(self,):
        super(HeatmapLossMSEmask, self).__init__()

    def masked_mse(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE averaged only over the gt != 0 positions.

        Args:
            pred (B, C, H, W)
            gt   (B, C, H, W)
        Returns:
            scalar loss
        """
        # 1) compute per-pixel squared error
        se = (pred - gt).pow(2)

        # 2) build a mask of where GT is non-zero
        mask = (gt != 0).float()          # same shape as se

        # 3) apply mask, sum up, and divide by number of valid pixels
        valid_count = mask.sum()
        loss = se.mul(mask).sum() / (valid_count + self.eps)
        return loss


    def forward(self, heatmap_preds, heatmap_gt):
        
        # Handle batched input with multiple frames if needed
        if len(heatmap_gt.size()) > 4:
            bs, fr, c, h, w = heatmap_gt.size()
            heatmap_gt = heatmap_gt.view(-1, c, h, w)

        heatmap_loss = self.masked_mse(heatmap_preds, heatmap_gt)
        return heatmap_loss



    
class StageHeatmapLossMSE(nn.Module):
    def __init__(self,):
        """
        MSE For many stages
        """
        super(StageHeatmapLossMSE, self).__init__()

    def mse_loss(self, pred, gt):
        """
        Compute MSE loss between prediction and ground truth.
        
        Args:
            pred (tensor): Predicted heatmap. Shape: (B, C, H, W).
            gt (tensor): Ground truth heatmap. Shape: (B, C, H, W).
        
        Returns:
            loss (tensor): MSE loss computed with a sum reduction.
        """
        return F.mse_loss(pred, gt, reduction='mean')

    def forward(self, heatmap_preds, heatmap_gt):
        
        # Handle batched input with multiple frames if needed
        if len(heatmap_gt.size()) > 4:
            bs, fr, c, h, w = heatmap_gt.size()
            heatmap_gt = heatmap_gt.view(-1, c, h, w)

        total = 0.0
        for pred in heatmap_preds:
            total +=  self.mse_loss(F.sigmoid(pred), heatmap_gt)

        return total
    

    
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

