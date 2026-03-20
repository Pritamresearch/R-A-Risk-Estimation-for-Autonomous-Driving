
"""# UPDATED: config/config.py
# Added lists for experiments + cleaned CLASS_WEIGHTS + DATASETS paths (from your main.py comments)

class Config:
    DEVICE = "cuda"  # change to "cpu" if no GPU

    # === EXPERIMENT PARAMETERS ===
    LAMBDA_VALUES = [0.15, 0.05, 0.10, 0.25]      # all 4 values you added
    FORMULAS = ["exponential", "inverse", "linear"]

    # === OTHER SETTINGS (your latest values) ===
    REGION_WEIGHTS = [0.4, 0.6, 0.8, 1.0]
    CLASS_WEIGHTS = {
        0: 0.00,   # background
        1: 1.30,   # person
        2: 1.10,   # bicycle
        3: 1.20,   # car
        4: 1.10,   # motorcycle
        6: 1.00,   # bus
        17: 0.50,  # dog
        18: 0.50,  # cat
        11: 0.05,  # sky
        7: 0.05,   # road
    }

    # Video paths for the 3 datasets you mentioned
    DATASETS = {
        "real_life": r"D:\yolo with video sementic depth\data\WhatsApp Video 2026-03-10 at 12.43.54 AM.mp4",
        "kitti": r"D:\yolo with video sementic depth\data\kitti\testing\challenge_video.mp4",
        "bdd100k": r"D:\yolo with video sementic depth\data\bdd100k_videos_train_00\bdd100k\videos\train\00a0f008-3c67908e.mov",
    }

"""














class Config:

    DEVICE = "cuda"   # change to "cpu" if no GPU

    LAMBDA_DISTANCE = 0.25

    REGION_WEIGHTS = [0.4, 0.6, 0.8, 1.0]

    CLASS_WEIGHTS = {
        0:  0.00,   # background (may include sky/road if not separate)

        # High priority – people & vehicles
        1:  1.30,   # person           (COCO 0 → +1)
        2:  1.10,   # bicycle          (COCO 1)
        3:  1.20,   # car              (COCO 2)
        4:  1.10,   # motorcycle       (COCO 3)
        6:  1.00,   # bus              (COCO 5)


        # Lower – animals / potential sudden hazards
        17: 0.50,   # dog              (COCO 16)
        18: 0.50,   # cat              (COCO 17)

        # ────────────────────────────────────────────────
        # Sky and road added – usually very low risk
        # ────────────────────────────────────────────────
        11: 0.05,   # sky              (very low risk, far away)
        7:  0.05,   # road             (very low risk – road itself is not dangerous)
                    # Change to 0.0 if you want zero risk contribution from road pixels
    }