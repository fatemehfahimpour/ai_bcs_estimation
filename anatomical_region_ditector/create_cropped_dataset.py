from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm

# ============================================================
# Configuration & Directory Structure
# ============================================================

# CURRENT_DIR: .../ai_bcs_estimation/anatomical_region_ditector
CURRENT_DIR = Path(__file__).resolve().parent

# PROJECT_ROOT: .../ai_bcs_estimation
PROJECT_ROOT = CURRENT_DIR.parent

# Source and Destination Datasets located directly inside PROJECT_ROOT
SRC_DATASET_DIR = PROJECT_ROOT / "dataset"
DST_DATASET_DIR = PROJECT_ROOT / "cropped_dataset"

# Model weights located inside the detector folder
MODEL_WEIGHTS = CURRENT_DIR / "runs" / "detect" / "anatomical_region_yolo11m" / "weights" / "best.pt"

# Inference parameters
CONF_THRESHOLD = 0.25
IMAGE_SIZE = 640
BATCH_SIZE = 32
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Supported image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Selection Weights for Composite Scoring
W_CONF = 0.45  # Weight for detector confidence
W_AREA = 0.25  # Weight for normalized box area
W_CENTER = 0.30  # Weight for closeness to the image center


def select_best_box(boxes_xyxy, confidences, img_shape):
    """
    Selects the optimal bounding box using a composite score:
    Score = (w_conf * Conf) + (w_area * Area_norm) + (w_center * Center_closeness)
    """
    img_h, img_w = img_shape[:2]
    img_center_x = img_w / 2.0
    img_center_y = img_h / 2.0
    max_dist = np.hypot(img_center_x, img_center_y)  # Max Euclidean distance from center
    total_img_area = float(img_w * img_h)

    best_score = -1.0
    best_box = None

    for box, conf in zip(boxes_xyxy, confidences):
        x1, y1, x2, y2 = box
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        box_area = bw * bh

        # 1. Confidence component [0 to 1]
        score_conf = float(conf)

        # 2. Area component: normalized relative to image area [0 to 1]
        score_area = min(1.0, box_area / (total_img_area * 0.5))

        # 3. Center closeness component [0 to 1]
        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0
        dist_to_center = np.hypot(box_center_x - img_center_x, box_center_y - img_center_y)
        score_center = max(0.0, 1.0 - (dist_to_center / max_dist))

        # Composite score
        total_score = (W_CONF * score_conf) + (W_AREA * score_area) + (W_CENTER * score_center)

        if total_score > best_score:
            best_score = total_score
            best_box = box

    return best_box


def crop_image(img, box):
    """Crop image with coordinates safely clamped within image boundaries."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = map(int, box)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    return img[y1:y2, x1:x2]


def main():
    if not SRC_DATASET_DIR.exists():
        raise FileNotFoundError(f"Source dataset directory not found: {SRC_DATASET_DIR.resolve()}")

    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Model weights not found: {MODEL_WEIGHTS.resolve()}")

    print(f"Project root: {PROJECT_ROOT.resolve()}")
    print(f"Loading YOLO detector from: {MODEL_WEIGHTS.resolve()}")
    print(f"Using compute device: {DEVICE}")
    model = YOLO(str(MODEL_WEIGHTS))

    class_dirs = [d for d in SRC_DATASET_DIR.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(f"No subdirectories found in {SRC_DATASET_DIR.resolve()}")

    print(f"Found {len(class_dirs)} BCS class folders: {[d.name for d in class_dirs]}")

    total_images_processed = 0
    total_fallback_used = 0

    for cls_dir in sorted(class_dirs):
        class_name = cls_dir.name
        dst_class_dir = DST_DATASET_DIR / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)

        image_paths = [
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not image_paths:
            print(f"Skipping {class_name}: No images found.")
            continue

        print(f"\nProcessing class '{class_name}' ({len(image_paths)} images)...")

        for i in tqdm(range(0, len(image_paths), BATCH_SIZE), desc=f"Class {class_name}"):
            batch_paths = image_paths[i: i + BATCH_SIZE]

            results = model.predict(
                source=[str(p) for p in batch_paths],
                imgsz=IMAGE_SIZE,
                conf=CONF_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )

            for img_path, result in zip(batch_paths, results):
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"\nWarning: Could not read image: {img_path}")
                    continue

                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()

                    # Smart Selection (Confidence + Area + Center Closeness)
                    best_box = select_best_box(xyxy, confs, img.shape)
                    crop = crop_image(img, best_box)
                else:
                    # Fallback if no detection found
                    crop = img
                    total_fallback_used += 1

                out_path = dst_class_dir / img_path.name
                cv2.imwrite(str(out_path), crop)
                total_images_processed += 1

    print("\n" + "=" * 50)
    print("Smart Cropping Dataset Pipeline Completed Successfully!")
    print(f"Total images cropped and saved: {total_images_processed}")
    print(f"Fallback full images (no bbox detected): {total_fallback_used}")
    print(f"Cropped dataset location: {DST_DATASET_DIR.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
