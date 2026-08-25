from pathlib import Path

import torch
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================

DATA_YAML = "data_detector.yaml"

# مدل قابل تغییر است:
# yolo11n.pt
# yolo11s.pt
# yolo11m.pt
# yolo11l.pt
# yolo11x.pt
MODEL_WEIGHTS = "yolo11m.pt"

PROJECT_DIR = "runs/detect"
RUN_NAME = "anatomical_region_yolo11m"

EPOCHS = 200
IMAGE_SIZE = 768
BATCH_SIZE = 8
PATIENCE = 40

# برای GPU: 0
# برای CPU: "cpu"
DEVICE = 0

WORKERS = 4
SEED = 42

def main() -> None:
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"Could not find {DATA_YAML}."
            "Run prepare_detector_dataset.py first."
        )

    if DEVICE != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        device = "cpu"
    else:
        device = DEVICE

    print(f"Using device: {device}")

    model = YOLO(MODEL_WEIGHTS)

    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,
        seed=SEED,

        project=PROJECT_DIR, 
        name=RUN_NAME,
        exist_ok=True,

        #Checkpointing and stopping
        save=True,
        plots=True,
        patience=PATIENCE,
        val=True,

        # Geometric augmentation
        degrees=8.0,
        translate=0.08,
        scale=0.35,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,

        # Photometric augmentation
        # These help with illumination, shadows and low contrast.
        hsv_h=0.01,
        hsv_s=0.35,
        hsv_v=0.35,

         # Mosaic can help generalization, but close it near the end.
        mosaic=0.7,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=15,

        # Deterministic behavior
        deterministic=True,
        amp=True,
    )

    best_model = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"

    print("\nTraining completed.")
    print(f"Best model: {best_model.resolve()}")


if __name__ == "__main__":
    main()