import torch
import torch.nn as nn
from depth_anything_3.api import DepthAnything3
from PIL import Image  # For converting numpy to PIL if needed
import numpy as np


class DepthModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = DepthAnything3.from_pretrained(
            "depth-anything/da3-base"  # Or "da3-large" for larger model
        )

    def forward(self, image):

        # Convert numpy BGR/RGB to PIL if needed (model expects PIL or path)
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Inference (DA3 takes list of images)
        prediction = self.model.inference([image])

        depth = prediction.depth  # [H, W] float32

        # Normalize as in your original code
        depth = depth / (depth.max() + 1e-8)

        # Convert back to torch tensor if needed downstream
        depth = torch.from_numpy(depth).unsqueeze(0).to(self.model.device)  # Add batch dim and move to device

        return depth