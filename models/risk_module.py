"""# UPDATED: models/risk_module.py
# This implements ALL THREE risk formulas with your requested changes.
# - exponential: original (uses lambda_d)
# - inverse: weights / depth
# - linear: weights * clamp(1 - lambda_d * depth, 0)  (reuses lambda_d as decay rate for consistency)
# Also keeps all your device/batch/resize improvements.

import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskEstimation(nn.Module):
    def __init__(self, class_weights, lambda_d, formula="exponential", device="cpu"):
        super().__init__()
        self.lambda_d = lambda_d
        self.formula = formula
        self.device = torch.device(device)

        # Build weight tensor (same as your version)
        max_class = max(class_weights.keys(), default=80) + 1
        weight_tensor = torch.zeros(max_class, dtype=torch.float32, device=self.device)
        for cid, w in class_weights.items():
            if 0 <= cid < max_class:
                weight_tensor[cid] = float(w)
        self.register_buffer("weight_tensor", weight_tensor)

    def forward(self, seg, depth):
        # Move to device + handle missing batch dim (your logic)
        seg = seg.to(self.device, dtype=torch.long, non_blocking=True)
        depth = depth.to(self.device, dtype=torch.float32, non_blocking=True)

        if seg.dim() == 2:
            seg = seg.unsqueeze(0)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)

        # Clamp seg ids
        seg = torch.clamp(seg, 0, self.weight_tensor.shape[0] - 1)
        weights = self.weight_tensor[seg]  # (B, H, W)

        # Resize depth to match seg (your improved logic)
        if depth.shape[-2:] != seg.shape[-2:]:
            target_size = seg.shape[-2:]
            if depth.dim() == 3:  # (B, H_d, W_d)
                depth = depth.unsqueeze(1)  # → (B, 1, H_d, W_d)
            depth = F.interpolate(
                depth, size=target_size, mode="bilinear", align_corners=False
            )
            if depth.shape[1] == 1:
                depth = depth.squeeze(1)

        # ==================== THREE FORMULAS ====================
        depth = depth.clamp(min=1e-6)

        if self.formula == "exponential":
            risk = weights * torch.exp(-self.lambda_d * depth)
        elif self.formula == "inverse":
            risk = weights / depth
        elif self.formula == "linear":
            risk = weights * torch.clamp(1.0 - self.lambda_d * depth, min=0.0)
        else:
            raise ValueError(f"Unknown formula: {self.formula}. Use 'exponential', 'inverse' or 'linear'.")

        # Optional: remove batch dim for consistency with original code
        if risk.shape[0] == 1:
            risk = risk.squeeze(0)

        return risk
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class RiskEstimation(nn.Module):

    def __init__(self, class_weights, lambda_d, device="cpu"):
        super().__init__()
        self.lambda_d = lambda_d
        self.device = torch.device(device)

        max_class = max(class_weights.keys(), default=80) + 1
        weight_tensor = torch.zeros(max_class, dtype=torch.float32, device=self.device)

        for cid, w in class_weights.items():
            if 0 <= cid < max_class:
                weight_tensor[cid] = float(w)

        self.register_buffer("weight_tensor", weight_tensor)

    def forward(self, seg, depth):
        seg   = seg.to(self.device, dtype=torch.long,   non_blocking=True)
        depth = depth.to(self.device, dtype=torch.float32, non_blocking=True)

        # Handle possible missing batch dim
        if seg.dim() == 2:
            seg = seg.unsqueeze(0)
        if depth.dim() == 2:
            depth = depth.unsqueeze(0)

        seg = torch.clamp(seg, 0, self.weight_tensor.shape[0] - 1)
        weights = self.weight_tensor[seg]  # (B, H_seg, W_seg)

        # Resize depth to match seg spatial size
        if depth.shape[-2:] != seg.shape[-2:]:
            target_size = seg.shape[-2:]

            # Add channel dim if missing
            if depth.dim() == 3:  # (B, H_d, W_d)
                depth = depth.unsqueeze(1)  # → (B, 1, H_d, W_d)

            depth = F.interpolate(
                depth,
                size=target_size,
                mode="bilinear",
                align_corners=False
            )

            # Squeeze channel if added
            if depth.shape[1] == 1:
                depth = depth.squeeze(1)

        # depth and weight both
        #risk = weights * torch.exp(-self.lambda_d * depth.clamp(min=1e-6))

        # only depth
        #risk = weights * torch.exp( depth.clamp(min=1e-6))

        #only weight
        risk = weights

        if risk.shape[0] == 1:
            risk = risk.squeeze(0)

        return risk