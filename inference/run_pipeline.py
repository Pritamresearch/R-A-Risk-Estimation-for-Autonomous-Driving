"""import cv2
import torch
import numpy as np
import time
import logging
import csv
import os
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

# ─── Suppress unwanted logs ───
logging.getLogger("transformers").setLevel(logging.WARNING)

from models.risk_module import RiskEstimation
from models.region_module import RegionDecision
from config.config import Config


def run_pipeline(video_path, seg_model, depth_model, device,
                 formula="exponential", lambda_d=0.15, output_dir=None):
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    # === RISK MODULE with chosen formula & lambda (your updated version) ===
    risk_module = RiskEstimation(
        Config.CLASS_WEIGHTS,
        lambda_d=lambda_d,
        formula=formula,
        device=device
    ).to(device)

    region_module = RegionDecision(Config.REGION_WEIGHTS)

    # === SEPARATE OUTPUT FOLDER FOR THIS COMBINATION ===
    if output_dir is None:
        output_dir = "default_output"
    os.makedirs(output_dir, exist_ok=True)

    # ─── Your original settings ───
    RECORD_OUTPUT = False          # Change to True if you want video
    SAVE_FRAMES = True

    out = None
    out_path = os.path.join(output_dir, "risk_visualization_output.mp4")
    FRAME_SAVE_DIR = os.path.join(output_dir, "saved_frames")
    CSV_PATH = os.path.join(output_dir, "risk_metrics_log.csv")

    if SAVE_FRAMES:
        os.makedirs(FRAME_SAVE_DIR, exist_ok=True)
    if RECORD_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # ─── CSV logging ───
    csv_file_exists = os.path.exists(CSV_PATH)
    csv_headers = ["frame", "timestamp", "fps", "decision", "mean_risk",
                   "max_risk", "collision_risk", "has_close_high_risk"]
    csvfile = open(CSV_PATH, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    if not csv_file_exists:
        writer.writeheader()

    print(f"\n Running → Dataset: {os.path.basename(video_path)} | "
          f"Formula: {formula} | Lambda: {lambda_d}")
    print(f"   All outputs saved to: {output_dir}")
    print("Press 'q' to quit\n")

    prev_time = time.time()
    frame_count = 0

    CLOSE_DEPTH_THRESHOLD = 0.25
    HIGH_RISK_CLASSES = {1, 2, 3, 4, 6}
    ROAD_CLASS = 7
    COLLISION_THRESHOLD = 0.6

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        frame = cv2.resize(frame, (640, 360))
        frame_count += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            seg = seg_model(rgb)
            depth = depth_model(rgb)
            risk = risk_module(seg, depth)

        # ────────────────────────────────────────────────
        # Prepare numpy arrays (your exact code)
        # ────────────────────────────────────────────────
        rgb_np = rgb.copy()
        seg_np = seg.squeeze().cpu().numpy().astype(np.uint8)
        depth_np = depth.squeeze().cpu().numpy()
        risk_np = risk.squeeze().cpu().numpy()

        target_h, target_w = rgb_np.shape[:2]
        seg_np = cv2.resize(seg_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        depth_np = cv2.resize(depth_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        risk_np = cv2.resize(risk_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # ─── Road mask for filtering ───
        road_mask = (seg_np == ROAD_CLASS).astype(np.float32)
        risk_road_only = risk_np * road_mask
        risk_torch = torch.from_numpy(risk_road_only).to(device)
        decision, urgency = region_module.compute(risk_torch)

        # ─── Segmentation visualization (your exact colors) ───
        seg_vis = np.zeros((seg_np.shape[0], seg_np.shape[1], 3), dtype=np.uint8)
        road_pixels = (seg_np == ROAD_CLASS)
        seg_vis[road_pixels] = [0, 255, 0]           # Road GREEN
        seg_vis[seg_np == 1] = [0, 0, 255]           # Person RED
        seg_vis[seg_np == 2] = [255, 255, 0]         # Bicycle CYAN
        seg_vis[seg_np == 3] = [255, 0, 0]           # Car BLUE
        seg_vis[seg_np == 4] = [255, 0, 255]         # Motorcycle MAGENTA
        seg_vis[seg_np == 6] = [0, 165, 255]         # Bus ORANGE

        other_mask = ~(road_pixels | (seg_np == 1) | (seg_np == 2) |
                       (seg_np == 3) | (seg_np == 4) | (seg_np == 6))
        if other_mask.any():
            other_input = (seg_np[other_mask] % 20 * 12).astype(np.uint8)
            other_color = cv2.applyColorMap(other_input, cv2.COLORMAP_TURBO).squeeze(1)
            seg_vis[other_mask] = other_color

        seg_color = seg_vis

        # ─── Colormaps ───
        risk_norm = (risk_np - risk_np.min()) / (risk_np.max() - risk_np.min() + 1e-8)
        depth_norm = depth_np / (depth_np.max() + 1e-8)
        risk_color = cv2.applyColorMap((risk_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # ─── 4 Panels ───
        panel_size = (480, 360)
        rgb_small = cv2.resize(rgb_np, panel_size)
        seg_small = cv2.resize(seg_color, panel_size, interpolation=cv2.INTER_NEAREST)
        depth_small = cv2.resize(depth_color, panel_size)
        risk_small = cv2.resize(risk_color, panel_size)

        cv2.putText(rgb_small, "RGB", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
        cv2.putText(seg_small, "SEG", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
        cv2.putText(depth_small, "DEPTH", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
        cv2.putText(risk_small, "RISK", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)

        # ─── Close objects stats (your exact code) ───
        close_mask = depth_np < CLOSE_DEPTH_THRESHOLD
        unique_classes = np.unique(seg_np)
        detected = [c for c in unique_classes if c != 0 and c != ROAD_CLASS]
        close_stats = []
        has_close_high_risk = False

        for cls in detected:
            class_mask = (seg_np == cls) & close_mask
            if class_mask.any():
                depths = depth_np[class_mask]
                if len(depths) > 0:
                    avg = np.mean(depths)
                    min_d = np.min(depths)
                    count = np.sum(class_mask)
                    name = f"Cls {cls}"
                    if cls == 1: name = "Person"
                    elif cls == 2: name = "Cyclist"
                    elif cls == 3: name = "Car"
                    elif cls == 4: name = "Motorcycle"
                    elif cls == 6: name = "Bus"
                    line = f"{name} ({count} px): Avg {avg:.2f} Min {min_d:.2f}"
                    close_stats.append((avg, line, cls))
                    if cls in HIGH_RISK_CLASSES:
                        has_close_high_risk = True

        close_stats.sort(key=lambda x: x[0])
        close_text = "\n".join([line for _, line, _ in close_stats[:5]]) if close_stats else "No close objects detected"
        close_color = (0, 255, 0) if close_stats else (0, 200, 255)

        # Add text to panels
        y0, dy = 70, 25
        for i, line in enumerate(close_text.split("\n")):
            y = y0 + i * dy
            cv2.putText(depth_small, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, close_color, 1)
            cv2.putText(risk_small, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, close_color, 1)

        # ─── Combine panels ───
        top = np.hstack([rgb_small, seg_small])
        bottom = np.hstack([depth_small, risk_small])
        combined = np.vstack([top, bottom])

        # ─── Road straightness + instruction (your code) ───
        h, w = road_mask.shape
        lower_road = road_mask[h//2:, :]
        road_visible = lower_road.sum() > 100
        road_straight = False
        if road_visible:
            col_sum = lower_road.sum(axis=0)
            center_strength = col_sum[w//3:2*w//3].sum()
            side_strength = col_sum[:w//3].sum() + col_sum[2*w//3:].sum()
            road_straight = center_strength > side_strength * 2.5

        if has_close_high_risk:
            instruction = "BRAKE NOW, close danger on road!"
            instr_color = (0, 0, 255)
        elif road_straight and not has_close_high_risk:
            instruction = "Road straight & clear , INCREASE SPEED"
            instr_color = (0, 255, 0)
        elif road_straight:
            instruction = "Road straight , proceed carefully"
            instr_color = (0, 200, 255)
        else:
            instruction = "Road curving , keep safe speed"
            instr_color = (0, 165, 255)

        cv2.putText(combined, instruction, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, instr_color, 3)

        # ─── FPS + info ───
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time
        info = f"FPS: {fps:.1f} Decision: {decision.item()} Frame: {frame_count}"
        cv2.putText(combined, info, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # ─── CSV logging (your exact metrics) ───
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        danger_pixels = (risk_np > COLLISION_THRESHOLD).sum()
        collision_risk = danger_pixels / (risk_np.size + 1e-8)

        metrics_row = {
            "frame": frame_count,
            "timestamp": current_time_str,
            "fps": round(fps, 2),
            "decision": int(decision.item()),
            "mean_risk": round(float(risk_np.mean()), 4),
            "max_risk": round(float(risk_np.max()), 4),
            "collision_risk": round(collision_risk, 4),
            "has_close_high_risk": 1 if has_close_high_risk else 0
        }
        writer.writerow(metrics_row)
        csvfile.flush()

        # ─── Display & save ───
        cv2.imshow("Realtime Semantic Risk – 4 Panels", combined)
        if SAVE_FRAMES:
            cv2.imwrite(os.path.join(FRAME_SAVE_DIR, f"frame_{frame_count:06d}.png"), combined)
        if RECORD_OUTPUT:
            if out is None:
                h, w = combined.shape[:2]
                out = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
            out.write(combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ─── Cleanup ───
    cap.release()
    if out is not None:
        out.release()
    csvfile.close()
    cv2.destroyAllWindows()

    print(f" Experiment finished → {frame_count} frames | CSV: {CSV_PATH}")

    # ─── AUTO PLOT (your exact plot, saved in same folder) ───
    try:
        df = pd.read_csv(CSV_PATH)
        if not df.empty:
            plt.figure(figsize=(14, 7))
            plt.plot(df['frame'], df['mean_risk'], label='Mean Risk', color='blue', linewidth=1.4)
            plt.plot(df['frame'], df['collision_risk'], label='Collision Risk', color='orange', linewidth=1.4)
            plt.plot(df['frame'], df['max_risk'], label='Max Risk', color='red', linewidth=1.8)
            plt.xlabel('Frame Number')
            plt.ylabel('Risk Value')
            plt.title(f'Risk Metrics Evolution\nFormula: {formula} | λ = {lambda_d} | {os.path.basename(video_path)}')
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"risk_plot_{formula}_lambda{lambda_d:.2f}.png")
            plt.savefig(plot_path, dpi=200)
            plt.close()
            print(f"📊 Plot saved → {plot_path}")
    except Exception as e:
        print(f"Plotting skipped: {e}")"""

import cv2
import torch
import numpy as np
import time
import logging
import csv
import os
from datetime import datetime

# ─── Suppress unwanted Hugging Face / transformers logs ───
logging.getLogger("transformers").setLevel(logging.WARNING)

from models.risk_module import RiskEstimation
from models.region_module import RegionDecision
from config.config import Config


def run_pipeline(video_path, seg_model, depth_model, device):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video file: {video_path}")
        return

    risk_module = RiskEstimation(
        Config.CLASS_WEIGHTS,
        Config.LAMBDA_DISTANCE,
        device=device
    ).to(device)

    region_module = RegionDecision(Config.REGION_WEIGHTS)

    # ─── Video saving option ───
    RECORD_OUTPUT = False           # ← Change to True to save continuous video
    out = None
    out_path = "risk_visualization_output.mp4"
    if RECORD_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"Will save output video to: {out_path}")

    # ─── Frame-wise image saving option ───
    SAVE_FRAMES = True              # ← Change to False when you don't need individual images
    FRAME_SAVE_DIR = "real_life_saved_frames_only_weight_exp_0.25"
    FRAME_FORMAT = "png"            # can be "jpg" if you prefer smaller files

    if SAVE_FRAMES:
        os.makedirs(FRAME_SAVE_DIR, exist_ok=True)
        print(f"Will save individual frames to folder: {FRAME_SAVE_DIR}")

    # ─── CSV logging setup (risk metrics) ───
    CSV_PATH = "real_life_risk_metrics_log_only_weight_exp_0.25.csv"
    csv_file_exists = os.path.exists(CSV_PATH)

    csv_headers = [
        "frame",
        "timestamp",
        "fps",
        "decision",
        "mean_risk",
        "max_risk",
        "collision_risk",
        "has_close_high_risk"
    ]

    csvfile = open(CSV_PATH, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)

    if not csv_file_exists:
        writer.writeheader()
        print(f"Created new risk metrics log: {CSV_PATH}")
    else:
        print(f"Appending to existing risk metrics log: {CSV_PATH}")

    print("Starting video playback with 4-panel visualization...")
    print("Press 'q' to quit")

    prev_time = time.time()
    frame_count = 0

    CLOSE_DEPTH_THRESHOLD = 0.25
    HIGH_RISK_CLASSES = {1, 2, 3, 4, 6}   # person, bicycle, car, motorcycle, bus
    ROAD_CLASS = 7                        # change if road class is different in your ADE20K mapping
    COLLISION_THRESHOLD = 0.6

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        frame = cv2.resize(frame, (640, 360))
        frame_count += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with torch.no_grad():
            seg   = seg_model(rgb)
            depth = depth_model(rgb)
            risk  = risk_module(seg, depth)

        # ────────────────────────────────────────────────
        # Prepare numpy arrays
        # ────────────────────────────────────────────────
        rgb_np   = rgb.copy()
        seg_np   = seg.squeeze().cpu().numpy().astype(np.uint8)
        depth_np = depth.squeeze().cpu().numpy()
        risk_np  = risk.squeeze().cpu().numpy()

        target_h, target_w = rgb_np.shape[:2]
        seg_np   = cv2.resize(seg_np,   (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        depth_np = cv2.resize(depth_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        risk_np  = cv2.resize(risk_np,  (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # ────────────────────────────────────────────────
        # Road mask for filtering decision & visualization
        # ────────────────────────────────────────────────
        road_mask = (seg_np == ROAD_CLASS).astype(np.float32)

        # Risk only on road pixels
        risk_road_only = risk_np * road_mask
        risk_torch = torch.from_numpy(risk_road_only).to(device)
        decision, urgency = region_module.compute(risk_torch)

        # ────────────────────────────────────────────────
        # Segmentation visualization – road = GREEN + distinct object colors
        # ────────────────────────────────────────────────
        seg_vis = np.zeros((seg_np.shape[0], seg_np.shape[1], 3), dtype=np.uint8)

        # Road – GREEN
        road_pixels = (seg_np == ROAD_CLASS)
        seg_vis[road_pixels] = [0, 255, 0]

        # Person – RED
        mask = (seg_np == 1)
        seg_vis[mask] = [0, 0, 255]

        # Bicycle – CYAN
        mask = (seg_np == 2)
        seg_vis[mask] = [255, 255, 0]

        # Car – BLUE
        mask = (seg_np == 3)
        seg_vis[mask] = [255, 0, 0]

        # Motorcycle – MAGENTA
        mask = (seg_np == 4)
        seg_vis[mask] = [255, 0, 255]

        # Bus – ORANGE
        mask = (seg_np == 6)
        seg_vis[mask] = [0, 165, 255]

        # Other classes – TURBO colormap
        other_mask = ~(
            road_pixels |
            (seg_np == 1) | (seg_np == 2) | (seg_np == 3) |
            (seg_np == 4) | (seg_np == 6)
        )
        if other_mask.any():
            other_input = (seg_np[other_mask] % 20 * 12).astype(np.uint8)
            other_color = cv2.applyColorMap(other_input, cv2.COLORMAP_TURBO).squeeze(1)
            seg_vis[other_mask] = other_color

        seg_color = seg_vis

        # Normalize for colormaps
        risk_norm  = (risk_np - risk_np.min()) / (risk_np.max() - risk_np.min() + 1e-8)
        depth_norm = depth_np / (depth_np.max() + 1e-8)

        risk_color  = cv2.applyColorMap((risk_norm  * 255).astype(np.uint8), cv2.COLORMAP_JET)
        depth_color = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

        # ────────────────────────────────────────────────
        # Panels
        # ────────────────────────────────────────────────
        panel_size = (480, 360)

        rgb_small   = cv2.resize(rgb_np,   panel_size)
        seg_small   = cv2.resize(seg_color, panel_size, interpolation=cv2.INTER_NEAREST)
        depth_small = cv2.resize(depth_color, panel_size)
        risk_small  = cv2.resize(risk_color, panel_size)

        cv2.putText(rgb_small,   "RGB",   (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
        cv2.putText(seg_small,   "SEG",   (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
        cv2.putText(depth_small, "DEPTH", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
        cv2.putText(risk_small,  "RISK",  (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)

        # ────────────────────────────────────────────────
        # Close objects stats (with on-road filter option)
        # ────────────────────────────────────────────────
        close_mask = depth_np < CLOSE_DEPTH_THRESHOLD
        # close_mask = close_mask & road_mask.astype(bool)   # ← uncomment to filter only road

        unique_classes = np.unique(seg_np)
        detected = [c for c in unique_classes if c != 0 and c != ROAD_CLASS]

        close_stats = []
        has_close_high_risk = False

        for cls in detected:
            class_mask = (seg_np == cls) & close_mask
            if class_mask.any():
                depths = depth_np[class_mask]
                if len(depths) > 0:
                    avg   = np.mean(depths)
                    min_d = np.min(depths)
                    count = np.sum(class_mask)

                    name = f"Cls {cls}"
                    if cls == 1: name = "Person"
                    elif cls == 2: name = "Cyclist"
                    elif cls == 3: name = "Car"
                    elif cls == 4: name = "Motorcycle"
                    elif cls == 6: name = "Bus"

                    line = f"{name} ({count} px): Avg {avg:.2f}  Min {min_d:.2f}"
                    close_stats.append((avg, line, cls))

                    if cls in HIGH_RISK_CLASSES:
                        has_close_high_risk = True

        close_stats.sort(key=lambda x: x[0])

        if close_stats:
            lines = [line for _, line, _ in close_stats[:5]]
            close_text = "\n".join(lines)
            close_color = (0, 255, 0)
        else:
            close_text = "No close objects detected"
            close_color = (0, 200, 255)

        # Debug print when no close objects
        if not close_stats:
            print(f"Frame {frame_count} | Close pixels: {close_mask.sum()} | Detected classes: {detected}")

        # Add close stats to depth & risk panels
        y0, dy = 70, 25
        for i, line in enumerate(close_text.split("\n")):
            y = y0 + i * dy
            cv2.putText(depth_small, line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, close_color, 1)
            cv2.putText(risk_small,  line, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, close_color, 1)

        # ────────────────────────────────────────────────
        # Combine panels
        # ────────────────────────────────────────────────
        top    = np.hstack([rgb_small, seg_small])
        bottom = np.hstack([depth_small, risk_small])
        combined = np.vstack([top, bottom])

        # ────────────────────────────────────────────────
        # Road straightness check
        # ────────────────────────────────────────────────
        h, w = road_mask.shape
        lower_road = road_mask[h//2:, :]
        road_visible = lower_road.sum() > 100

        road_straight = False
        if road_visible:
            col_sum = lower_road.sum(axis=0)
            center_strength = col_sum[w//3:2*w//3].sum()
            side_strength   = col_sum[:w//3].sum() + col_sum[2*w//3:].sum()
            road_straight = center_strength > side_strength * 2.5

        # ────────────────────────────────────────────────
        # Driving instruction logic
        # ────────────────────────────────────────────────
        if has_close_high_risk:
            instruction = "BRAKE NOW, close danger on road!"
            instr_color = (0, 0, 255)
        elif road_straight and not has_close_high_risk:
            instruction = "Road straight & clear , INCREASE SPEED"
            instr_color = (0, 255, 0)
        elif road_straight:
            instruction = "Road straight , proceed carefully"
            instr_color = (0, 200, 255)
        else:
            instruction = "Road curving , keep safe speed"
            instr_color = (0, 165, 255)

        cv2.putText(combined, instruction, (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, instr_color, 3)

        # FPS + decision info
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
        prev_time = current_time

        info = f"FPS: {fps:.1f}   Decision: {decision.item()}   Frame: {frame_count}"
        cv2.putText(combined, info, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # ─── Risk metrics logging ───
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        danger_pixels = (risk_np > COLLISION_THRESHOLD).sum()
        collision_risk = danger_pixels / (risk_np.size + 1e-8)

        metrics_row = {
            "frame": frame_count,
            "timestamp": current_time_str,
            "fps": round(fps, 2),
            "decision": int(decision.item()),
            "mean_risk": round(float(risk_np.mean()), 4),
            "max_risk": round(float(risk_np.max()), 4),
            "collision_risk": round(collision_risk, 4),
            "has_close_high_risk": 1 if has_close_high_risk else 0
        }

        writer.writerow(metrics_row)
        csvfile.flush()

        # ─── Display ───
        cv2.imshow("Realtime Semantic Risk – 4 Panels", combined)

        # ─── Save individual frame if enabled ───
        if SAVE_FRAMES:
            frame_filename = f"frame_{frame_count:06d}.{FRAME_FORMAT}"
            frame_path = os.path.join(FRAME_SAVE_DIR, frame_filename)
            cv2.imwrite(frame_path, combined)

        # ─── Optional video recording ───
        if RECORD_OUTPUT:
            if out is None:
                h, w = combined.shape[:2]
                out = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
                print(f"Video recording started → {out_path}")
            out.write(combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ─── Cleanup ───
    cap.release()

    if out is not None:
        out.release()
        print(f"Video saved: {out_path}")
    else:
        print("Video recording was disabled (RECORD_OUTPUT = False)")

    if SAVE_FRAMES:
        print(f"Individual frames saved: {frame_count} files in folder '{FRAME_SAVE_DIR}'")
    else:
        print("Frame saving was disabled (SAVE_FRAMES = False)")

    csvfile.close()
    print(f"Metrics log closed. {frame_count} frames recorded to {CSV_PATH}")

    cv2.destroyAllWindows()
    print("Visualization stopped.")

