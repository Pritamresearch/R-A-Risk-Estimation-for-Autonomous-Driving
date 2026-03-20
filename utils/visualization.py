import cv2
import numpy as np


def visualize_frame(frame,seg,depth,risk,decision):

    seg = seg.squeeze().cpu().numpy()
    depth = depth.squeeze().cpu().numpy()
    risk = risk.squeeze().cpu().numpy()

    risk_norm = (risk - risk.min())/(risk.max()-risk.min()+1e-8)

    risk_map = (risk_norm*255).astype(np.uint8)

    risk_color = cv2.applyColorMap(risk_map,cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(frame,0.6,risk_color,0.4,0)

    cv2.putText(
        overlay,
        f"Decision:{decision}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,0,255),
        2
    )

    return overlay,risk_color