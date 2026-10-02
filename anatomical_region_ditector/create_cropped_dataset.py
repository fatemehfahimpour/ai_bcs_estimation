"""
Anatomical Region Extraction and Dataset Cropping Pipeline for Cow BCS Estimation.

This script runs batch inference using a fine-tuned YOLO11 detector to identify
the target anatomical region of cows from raw classification images. For images
with multiple candidate detections, it applies a multi-criteria composite scoring
strategy (detection confidence, relative bounding box area, and distance to image
center) to select the most relevant region of interest (ROI).

Directory Structure Expected:
    ai_bcs_estimation/
    ├── anatomical_region_detector/
    │   ├── crop_dataset.py (this script)
    │   └── runs/detect/anatomical_region_yolo11m/weights/best.pt
    ├── dataset/
    │   ├── 3.25/
    │   ├── 3.50/
    │   └── ...
    └── cropped_dataset/ (generated output directory)

Fallback Behavior:
    If no anatomical region is detected above the confidence threshold, the script
    preserves the original full image as a fallback to prevent data loss.
"""

from pathlib import Path
from typing import Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm
from ultralytics import YOLO

# ==============================================================================
# Configuration & Directory Structure
# ==============================================================================

# Directory anchors
CURRENT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = CURRENT_DIR.parent

# Source raw dataset and target cropped output directories
SRC_DATASET_DIR: Path = PROJECT_ROOT / "dataset"
DST_DATASET_DIR: Path = PROJECT_ROOT / "cropped_dataset"

# Pretrained YOLO detector weights
MODEL_WEIGHTS: Path = (
    CURRENT_DIR
    / "runs"
    / "detect"
    / "anatomical_region_yolo11m"
    / "weights"
    / "best.pt"
)

# Inference configuration
CONF_THRESHOLD: float = 0.25
IMAGE_SIZE: int = 640
BATCH_SIZE: int = 32
DEVICE: Union[int, str] = 0 if torch.cuda.is_available() else "cpu"

# Supported image file extensions
VALID_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Multi-criteria scoring weights for bounding box disambiguation
# Must sum to 1.0 for normalized interpretation
W_CONF: float = 0.45    # Detector prediction confidence
W_AREA: float = 0.25    # Normalized bounding box area relative to image size
W_CENTER: float = 0.30  # Proximity to image center (mitigates edge artifacts)


# ==============================================================================
# Helper Functions
# ==============================================================================

def select_best_box(
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    img_shape: Tuple[int, ...]
) -> np.ndarray:
    """Select the optimal bounding box using a composite multi-criteria score.

    The score balances detection confidence, bounding box area, and proximity
    to the center of the image:
        Score = (W_CONF * Conf) + (W_AREA * Area_norm) + (W_CENTER * Center_closeness)

    Args:
        boxes_xyxy: Bounding box coordinates of shape (N, 4) in [x1, y1, x2, y2] format.
        confidences: Detector confidence scores of shape (N,).
        img_shape: Dimensions of the input image (height, width, [channels]).

    Returns:
        np.ndarray: Coordinates of the highest-scoring bounding box [x1, y1, x2, y2].
    """
    img_h, img_w = img_shape[:2]
    img_center_x = img_w / 2.0
    img_center_y = img_h / 2.0

    # Maximum Euclidean distance from the center to any image corner
    max_dist = np.hypot(img_center_x, img_center_y)
    total_img_area = float(img_w * img_h)

    best_score = -1.0
    best_box = boxes_xyxy[0]

    for box, conf in zip(boxes_xyxy, confidences):
        x1, y1, x2, y2 = box
        bw = max(0.0, float(x2 - x1))
        bh = max(0.0, float(y2 - y1))
        box_area = bw * bh

        # 1. Confidence score [0.0, 1.0]
        score_conf = float(conf)

        # 2. Area score: normalized relative to 50% of the image canvas [0.0, 1.0]
        score_area = min(1.0, box_area / (total_img_area * 0.5))

        # 3. Center proximity score: inverted Euclidean distance [0.0, 1.0]
        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0
        dist_to_center = np.hypot(box_center_x - img_center_x, box_center_y - img_center_y)
        score_center = max(0.0, 1.0 - (dist_to_center / max_dist))

        # Weighted aggregate score
        total_score = (
            (W_CONF * score_conf)
            + (W_AREA * score_area)
            + (W_CENTER * score_center)
        )

        if total_score > best_score:
            best_score = total_score
            best_box = box

    return best_box


def crop_image(img: np.ndarray, box: Sequence[Union[int, float]]) -> np.ndarray:
    """Crop an image region safely clamping coordinates to image dimensions.

    Args:
        img: Input image as a NumPy array of shape (H, W, C).
        box: Coordinates [x1, y1, x2, y2] to crop.

    Returns:
        np.ndarray: Cropped sub-image array.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    # Restrict bounding box within the valid frame bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return img[y1:y2, x1:x2]


# ==============================================================================
# Pipeline Execution
# ==============================================================================

def main() -> None:
    """Execute the anatomical region detection and dataset cropping pipeline.

    Scans the source dataset for class folders, runs YOLO batch inference,
    crops the best detected anatomical region per image, and writes outputs
    to the destination directory.
    """
    # Validate required directories and weights
    if not SRC_DATASET_DIR.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {SRC_DATASET_DIR.resolve()}")

    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS.resolve()}")

    print(f"Project root: {PROJECT_ROOT.resolve()}")
    print(f"Loading YOLO detector from: {MODEL_WEIGHTS.resolve()}")
    print(f"Using compute device: {DEVICE}")

    # Load YOLO model
    model = YOLO(str(MODEL_WEIGHTS))

    # Retrieve all subdirectories (each corresponds to a BCS class label)
    class_dirs = [d for d in SRC_DATASET_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No subdirectories found in {SRC_DATASET_DIR.resolve()}")

    print(f"Found {len(class_dirs)} BCS class folders: {[d.name for d in class_dirs]}")

    total_images_processed = 0
    total_fallback_used = 0

    # Iterate through each class category
    for cls_dir in sorted(class_dirs):
        class_name = cls_dir.name
        dst_class_dir = DST_DATASET_DIR / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not image_paths:
            print(f"Skipping {class_name}: No valid images found.")
            continue

        print(f"\nProcessing class '{class_name}' ({len(image_paths)} images)...")

        # Process images in batches to optimize GPU utilization
        for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"Class {class_name}"):
            batch_paths = image_paths[i: i + BATCH_SIZE]

            # Run batch detector inference
            results = model.predict(
                source=[str(p) for p in batch_paths],
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )

            # Process individual results from the batch
            for img_path, result in zip(batch_paths, results):
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"\nWarning: Could not read image: {img_path}")
                    continue

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    # Select best candidate box and crop
                    best_box = select_best_box(xyxy, confs, img.shape)
                    crop = crop_image(img, best_box)
                else:
                    # Fallback to full image if no detection satisfies confidence threshold
                    crop = img
                    total_fallback_used += 1

                # Save cropped output image maintaining original filename
                out_path = dst_class_dir / img_path.name
                cv2.imwrite(str(out_path), crop)
                total_images_processed += 1

    # Print summary report
    print("\n" + "=" * 50)
    print("Smart Cropping Dataset Pipeline Completed Successfully!")
    print(f"Total images cropped and saved: {total_images_processed}")
    print(f"Fallback full images (no bbox detected): {total_fallback_used}")
    print(f"Cropped dataset location: {DST_DATASET_DIR.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
