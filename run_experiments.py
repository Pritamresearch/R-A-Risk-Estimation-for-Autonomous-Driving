import os
import torch
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

from config.config import Config
from models.segmentation_model import SegmentationModel
from models.depth_model import DepthModel
import inference.run_pipeline as rp   # ← this is your run_pipeline

# ====================== YOUR DATASETS ======================
DATASETS = [
    ("real",    r"D:\yolo with video sementic depth\data\WhatsApp Video 2026-03-10 at 12.43.54 AM.mp4"),
    ("kitti",   r"data\kitti\testing\challenge_video.mp4"),
    ("bdd100k", r"D:\yolo with video sementic depth\data\bdd100k_sample.mp4"),  # change if path wrong
]

FORMULAS = ["exp_decay", "inverse", "linear"]

BASE_OUTPUT_DIR = "experiment_results"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# ====================== YOUR EXACT PLOTTING CODE ======================
def plot_metrics(csv_path, save_to):
    df = pd.read_csv(csv_path)
    plt.figure(figsize=(14, 7))

    plt.plot(df['frame'], df['mean_risk'], 
             label='Mean Risk', color='blue', linewidth=1.4, alpha=0.9)

    plt.plot(df['frame'], df['collision_risk'], 
             label='Collision Risk', color='orange', linewidth=1.4, alpha=0.9)

    plt.plot(df['frame'], df['max_risk'], 
             label='Max Risk', color='red', linewidth=1.8, alpha=0.85)

    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Risk Value', fontsize=12)
    plt.title('Risk Metrics Evolution Over Time\n(mean, collision, and max risk)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(save_to, dpi=150)
    plt.close()
    print(f"   Plot saved → {os.path.basename(save_to)}")

# ====================== MAIN (EXPERIMENT) ======================
def main():
    # Same device logic as your main.py
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Using CPU (no CUDA detected)")

    print(f"Selected device: {device}\n")

    try:
        print("Loading Segmentation Model...")
        seg_model = SegmentationModel()

        print("Loading Depth Model...")
        depth_model = DepthModel().to(device)

        print("Creating Risk Estimation module...")
        # Risk module will be re-created inside run_pipeline with different formula/lambda

        print("\nStarting Full Experiment (3 formulas × 4 λ × 3 datasets = 36 runs)...")
        print("Press 'q' in any window to stop current run\n")

        run_counter = 0
        total_runs = len(DATASETS) * len(FORMULAS) * len(Config.LAMBDA_DISTANCES)

        for ds_name, video_path in DATASETS:
            if not os.path.exists(video_path):
                print(f"SKIP — Video not found: {video_path}")
                continue

            for formula in FORMULAS:
                for lam in Config.LAMBDA_DISTANCES:
                    run_counter += 1
                    print(f"\n[{run_counter}/{total_runs}] Running → {ds_name} | {formula} | λ = {lam:.2f}")

                    # Create unique folder for this combination
                    folder_name = f"{ds_name}__{formula}__lam{lam:.2f}_{datetime.now().strftime('%H%M%S')}"
                    out_dir = os.path.join(BASE_OUTPUT_DIR, folder_name)
                    os.makedirs(out_dir, exist_ok=True)

                    # Set globals so run_pipeline knows which formula and lambda to use
                    rp.CURRENT_FORMULA     = formula
                    rp.CURRENT_LAMBDA      = lam
                    rp.CURRENT_OUTPUT_DIR  = out_dir
                    rp.CURRENT_RUN_PREFIX  = f"{ds_name}__{formula}__lam{lam:.2f}__"

                    # Run your original pipeline logic
                    rp.run_pipeline(
                        video_path=video_path,
                        seg_model=seg_model,
                        depth_model=depth_model,
                        device=device
                    )

                    # Auto plot using your exact code
                    csv_file = os.path.join(out_dir, f"{rp.CURRENT_RUN_PREFIX}risk_metrics_log.csv")
                    plot_file = os.path.join(out_dir, "risk_plot.png")
                    if os.path.exists(csv_file):
                        plot_metrics(csv_file, plot_file)

        print("\n" + "="*70)
        print("🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
        print(f"Total runs: {run_counter}")
        print(f"Results saved in: {BASE_OUTPUT_DIR}")
        print("="*70)

    except Exception as e:
        print("\n" + "="*70)
        print("Pipeline crashed:")
        print(str(e))
        print("="*70)

if __name__ == "__main__":
    main()