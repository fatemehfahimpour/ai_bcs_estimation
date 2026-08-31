from pathlib import Path
import torch
from ultralytics import YOLO

# ============================================================
# Configuration
# ============================================================

# Resolve the directory where this script is located
CURRENT_DIR = Path(__file__).resolve().parent

# Path to the dataset YAML configuration file (located in the same directory)
DATA_YAML = str(CURRENT_DIR / "data_detector.yaml")

# Base directory for saving training runs and experiment results
PROJECT_DIR = str(CURRENT_DIR / "runs" / "detect")

# Experiment run identifier
RUN_NAME = "anatomical_region_yolo11m"

# Base pretrained weights (options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
MODEL_WEIGHTS = "yolo11m.pt"

# Training hyperparameters
EPOCHS = 1
IMAGE_SIZE = 416
BATCH_SIZE = 2
PATIENCE = 40  # Early stopping patience (number of epochs without improvement)

# Compute target: 0 (for GPU) or "cpu"
DEVICE = "cpu"

# Dataloader concurrency and random seed for reproducibility
WORKERS = 0
SEED = 42


def main() -> None:
    """Train the YOLO object detector on the prepared dataset.

    Validates prerequisite files, selects the compute device (GPU/CUDA with CPU fallback),
    initializes the YOLO model with pretrained weights, and executes the training loop
    with custom augmentations and deterministic behavior.

    Raises:
        FileNotFoundError: If the dataset YAML configuration file is missing.
    """
    # Verify dataset configuration exists before initiating training
    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(
            f"Could not find {DATA_YAML}. "
            "Run prepare_detector_dataset.py first."
        )

    # Validate GPU availability and handle fallback
    if DEVICE != "cpu" and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU.")
        device = "cpu"
    else:
        device = DEVICE

    print(f"Using device: {device}")

    # Load pretrained YOLO architecture
    model = YOLO(MODEL_WEIGHTS)

    # Execute training routine
    model.train(
        # Dataset and compute configuration
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=device,
        workers=WORKERS,
        seed=SEED,
        # Experiment tracking and logging
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        # Checkpointing and validation
        save=True,
        plots=True,
        patience=PATIENCE,
        val=True,
        # Geometric augmentations (simulates camera angles and perspective shifts)
        degrees=15.0,  # Image rotation (+/- deg)
        translate=0.08,  # Image translation (+/- fraction)
        scale=0.35,  # Image scale gain (+/- gain)
        shear=2.0,  # Shear angle (+/- deg)
        perspective=0.0005,  # Perspective distortion
        fliplr=0.5,  # Horizontal flip probability (exploits body bilateral symmetry)
        flipud=0.0,  # Vertical flip probability (disabled)
        # Photometric augmentations (handles lighting variations and harsh shadows)
        hsv_h=0.01,  # HSV-Hue augmentation (fraction)
        hsv_s=0.35,  # HSV-Saturation augmentation (fraction)
        hsv_v=0.35,  # HSV-Value/brightness augmentation (fraction)
        # Advanced composition augmentations (optional)
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=15,
        # Optimization and determinism
        deterministic=True,  # Guarantees reproducible training runs
        amp=True,  # Enables Automatic Mixed Precision (FP16) for speed and memory efficiency
    )

    # Display best checkpoint location
    best_model = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"

    print("\nTraining completed.")
    print(f"Best model: {best_model.resolve()}")


if __name__ == "__main__":
    main()
