import torch
import numpy as np
from ultralytics import YOLO

class SegmentationModel:

    def __init__(self):
        # Load model – Ultralytics auto-detects GPU if available
        self.model = YOLO("yolo26s-seg.pt")   # ← best balance (or "yolo26m-seg.pt" if you want more accuracy)

    def __call__(self, image):
        # Inference – force GPU if available, else CPU
        device = "0" if torch.cuda.is_available() else "cpu"  # "0" = first GPU
        results = self.model(image, verbose=False, device=device)

        h, w = image.shape[:2]
        seg = torch.zeros((h, w), dtype=torch.long)

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().long().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                seg[y1:y2, x1:x2] = clss[i] + 1  # shift COCO classes to 1-80

        return seg.unsqueeze(0)  # (1, H, W)