"""
Lightweight depth prediction head for LiDAR depth supervision.

Predicts a categorical depth distribution from FPN image features.
Used as an auxiliary loss to make backbone features depth-aware,
which improves BEV lifting in the BEVFormer encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthNet(nn.Module):
    """Categorical depth prediction head.

    Takes a single-scale FPN feature map and predicts per-pixel depth
    as a discrete distribution over depth bins.

    Args:
        in_channels (int): Input feature channels (from FPN). Default: 256.
        mid_channels (int): Hidden layer channels. Default: 256.
        num_bins (int): Number of depth bins. Default: 60.
        depth_min (float): Minimum depth in metres. Default: 1.0.
        depth_max (float): Maximum depth in metres. Default: 61.0.
        loss_weight (float): Weight for the depth loss. Default: 3.0.
        downsample (int): Downsample factor for GT depth map to match
            feature resolution. Default: 16.
    """

    def __init__(self,
                 in_channels=256,
                 mid_channels=256,
                 num_bins=60,
                 depth_min=1.0,
                 depth_max=61.0,
                 loss_weight=3.0,
                 downsample=16):
        super().__init__()
        self.num_bins = num_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.loss_weight = loss_weight
        self.downsample = downsample
        self.bin_size = (depth_max - depth_min) / num_bins

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, num_bins, 1),
        )

    def forward(self, feat):
        """Predict depth logits.

        Args:
            feat: (B*N, C, H, W) FPN feature map.

        Returns:
            (B*N, D, H, W) depth logits.
        """
        return self.net(feat)

    def loss(self, depth_logits, gt_depth):
        """Compute depth supervision loss.

        Args:
            depth_logits: (B*N, D, H_feat, W_feat) predicted depth logits.
            gt_depth: (B, N, H_img, W_img) sparse depth map from LiDAR
                projection. Zero means no valid depth.

        Returns:
            dict with 'loss_depth' scalar tensor.
        """
        if gt_depth is None:
            return dict(loss_depth=depth_logits.new_zeros(1).squeeze())

        B_N, D, H_feat, W_feat = depth_logits.shape

        # Flatten gt_depth to (B*N, 1, H_img, W_img)
        if gt_depth.dim() == 4:
            # (B, N, H, W)
            gt = gt_depth.reshape(-1, 1, gt_depth.shape[-2], gt_depth.shape[-1])
        elif gt_depth.dim() == 3:
            # (B*N, H, W)
            gt = gt_depth.unsqueeze(1)
        else:
            # (H, W) single sample
            gt = gt_depth.unsqueeze(0).unsqueeze(0)

        # Downsample GT to feature resolution using max-pool to preserve
        # valid depth values (sparse map, mostly zeros)
        gt_down = F.adaptive_max_pool2d(gt, (H_feat, W_feat))  # (B*N, 1, H_f, W_f)
        gt_down = gt_down.squeeze(1)  # (B*N, H_f, W_f)

        # Mask: only pixels with valid depth
        valid = (gt_down > self.depth_min) & (gt_down < self.depth_max)
        if valid.sum() == 0:
            return dict(loss_depth=depth_logits.new_zeros(1).squeeze())

        # Convert continuous depth to bin index
        bin_idx = ((gt_down - self.depth_min) / self.bin_size).long()
        bin_idx = bin_idx.clamp(0, self.num_bins - 1)  # (B*N, H_f, W_f)

        # One-hot target
        target = F.one_hot(bin_idx, self.num_bins)  # (B*N, H_f, W_f, D)
        target = target.permute(0, 3, 1, 2).float()  # (B*N, D, H_f, W_f)

        # Binary cross-entropy with logits, masked
        loss = F.binary_cross_entropy_with_logits(
            depth_logits, target, reduction='none')  # (B*N, D, H_f, W_f)

        # Apply mask: only supervise where LiDAR depth is valid
        valid_mask = valid.unsqueeze(1).expand_as(loss)  # (B*N, D, H_f, W_f)
        loss = (loss * valid_mask.float()).sum() / max(valid_mask.sum(), 1)

        return dict(loss_depth=loss * self.loss_weight)
