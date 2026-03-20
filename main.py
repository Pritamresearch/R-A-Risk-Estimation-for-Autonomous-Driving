"""# UPDATED: main.py
# Now runs ALL 36 combinations automatically (3 datasets × 4 λ × 3 formulas)
# Each combination gets its own clean folder with frames + CSV + plot.

import torch
import os
from inference.run_pipeline import run_pipeline
from models.segmentation_model import SegmentationModel
from models.depth_model import DepthModel
from config.config import Config


def main():
    # Device
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load models once (shared across all experiments)
    print("Loading Segmentation Model...")
    seg_model = SegmentationModel()
    # Optional: move model to GPU if you want (uncomment if needed)
    # seg_model.model = seg_model.model.to(device)

    print("Loading Depth Model...")
    depth_model = DepthModel().to(device)

    # === EXPERIMENT LOOP (your request) ===
    BASE_OUTPUT = "experiment_outputs"
    os.makedirs(BASE_OUTPUT, exist_ok=True)

    for ds_name, video_path in Config.DATASETS.items():
        for lambda_d in Config.LAMBDA_VALUES:
            for formula in Config.FORMULAS:
                combo_name = f"{ds_name}_lambda{lambda_d}_{formula}"
                output_dir = os.path.join(BASE_OUTPUT, combo_name)
                os.makedirs(output_dir, exist_ok=True)

                print(f"\n{'='*80}")
                print(f"RUNNING EXPERIMENT: {combo_name}")
                print(f"{'='*80}")

                run_pipeline(
                    video_path=video_path,
                    seg_model=seg_model,
                    depth_model=depth_model,
                    device=device,
                    formula=formula,
                    lambda_d=lambda_d,
                    output_dir=output_dir
                )

    print("\n🎉 ALL 36 EXPERIMENTS COMPLETED!")
    print(f"Check folder: {BASE_OUTPUT}")


if __name__ == "__main__":
    main()"""


import torch
from thop import profile

from inference.run_pipeline import run_pipeline
from models.segmentation_model import SegmentationModel
from models.depth_model import DepthModel
from models.risk_module import RiskEstimation      # needed if you use it here
from config.config import Config


def main():
    # Device selection (use GPU if available)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Using CPU (no CUDA detected)")

    print(f"Selected device: {device}")

    try:
        print("Loading Segmentation Model (YOLO26)...")
        seg_model = SegmentationModel()          # ← no device= here

        print("Loading Depth Model...")
        depth_model = DepthModel().to(device)

        # Risk module (pass device so weights & computations go to GPU)
        print("Creating Risk Estimation module...")
        risk_module = RiskEstimation(
            class_weights=Config.CLASS_WEIGHTS,
            lambda_d=Config.LAMBDA_DISTANCE,
            device=device                      # ← this is safe if you updated risk_module.py
        ).to(device)

        # Region decision is pure CPU/torch ops → no device needed
        from models.region_module import RegionDecision
        region_module = RegionDecision(Config.REGION_WEIGHTS)

        # new add for acc score
       

        print("\nStarting video pipeline...")
        print("Press 'q' to quit the windows\n")

        run_pipeline(
            #real life dataset
            video_path=r"D:\yolo with video sementic depth\data\WhatsApp Video 2026-03-10 at 12.43.54 AM.mp4",
            seg_model=seg_model,
            depth_model=depth_model,
            device=device
        )

    except Exception as e:
        print("\n" + "="*70)
        print("Pipeline crashed:")
        print(str(e))
        print("="*70)
       # import traceback
       # traceback.print_exc()



    
        print("\nQuick checks:")
        print("  • Video file really exists?")
        print("  • All tensors moved to same device?")
        print("  • Depth model output shape matches seg?")
        print("="*70)


if __name__ == "__main__":
    main()