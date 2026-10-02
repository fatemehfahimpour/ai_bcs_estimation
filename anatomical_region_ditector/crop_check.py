"""
Dataset Annotation Visualization and Cropping Inspection Tool for Cow BCS Estimation.

This script provides an interactive utility to inspect Pascal VOC-style XML or HTML
annotations alongside their corresponding raw images. It matches images with their
annotation files, allows interactive selection of samples and bounding boxes, and
visually displays the original image with the overlaid bounding box next to the
cropped region of interest (ROI).
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt

# Dataset directory path
DATASET_DIR: Path = Path("../dataset")


def parse_annotation(annotation_path: Path) -> List[Dict[str, Any]]:
    """Extract bounding box coordinates and labels from an XML or HTML annotation file (Pascal VOC format).

    Args:
        annotation_path: Path to the annotation file.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing object details.
    """
    try:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
    except ET.ParseError as error:
        print(f"Error reading annotation file: {annotation_path.name}")
        print(error)
        return []

    boxes: List[Dict[str, Any]] = []
    for obj_idx, obj in enumerate(root.findall("object"), start=1):
        name_elem = obj.find("name")
        bndbox = obj.find("bndbox")

        if bndbox is None:
            continue

        try:
            label = name_elem.text.strip() if name_elem is not None and name_elem.text else "Unknown"
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))

            boxes.append({
                "index": obj_idx,
                "label": label,
                "bbox": (xmin, ymin, xmax, ymax)
            })
        except (AttributeError, TypeError, ValueError):
            continue

    return boxes


def find_image_for_annotation(annotation_path: Path) -> Optional[Path]:
    """Find the corresponding image file for an annotation based on the file stem.

    Args:
        annotation_path: Path to the annotation file.

    Returns:
        Optional[Path]: Path to the matching image file if found, otherwise None.
    """
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    # 1. Check in the same directory where the annotation file is located
    for ext in image_extensions:
        img_path = annotation_path.with_suffix(ext)
        if img_path.exists():
            return img_path

    # 2. Check across all subdirectories of the dataset
    for img_path in DATASET_DIR.rglob("*"):
        if (
            img_path.is_file()
            and img_path.stem.lower() == annotation_path.stem.lower()
            and img_path.suffix.lower() in image_extensions
        ):
            return img_path

    return None


def get_all_pairs() -> List[Dict[str, Path]]:
    """Find all valid matched pairs of images and annotation files.

    Returns:
        List[Dict[str, Path]]: A list of dictionaries containing image and annotation paths.
    """
    annotation_files = sorted(
        [
            p for p in DATASET_DIR.rglob("*")
            if p.is_file() and p.suffix.lower() in [".xml", ".html"]
        ],
        key=lambda x: str(x).lower()
    )

    pairs: List[Dict[str, Path]] = []
    for ann_path in annotation_files:
        img_path = find_image_for_annotation(ann_path)
        if img_path is not None:
            pairs.append({
                "image_path": img_path,
                "annotation_path": ann_path
            })

    return pairs


def select_pair(pairs: List[Dict[str, Path]]) -> Optional[Dict[str, Path]]:
    """Display the list of pairs and receive the selected pair index from the user.

    Args:
        pairs: List of available image-annotation pairs.

    Returns:
        Optional[Dict[str, Path]]: Selected pair dictionary, or None if exiting.
    """
    print(f"\nTotal pairs found: {len(pairs)}\n")
    print("-" * 75)
    for idx, pair in enumerate(pairs, start=1):
        print(f"[{idx:3d}] Image: {pair['image_path'].name:<28} | Annotation: {pair['annotation_path'].name}")
    print("-" * 75)

    while True:
        choice = input(f"\nEnter the target pair index (1 to {len(pairs)} - or 'q' to quit): ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(pairs):
                return pairs[idx - 1]
        print(f"Invalid input! Please enter a number between 1 and {len(pairs)}.")


def select_box(boxes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Prompt the user to select a box index if there are multiple objects in the image.

    Args:
        boxes: List of parsed bounding boxes.

    Returns:
        Optional[Dict[str, Any]]: Selected box dictionary, or None if exiting.
    """
    if len(boxes) == 1:
        return boxes[0]

    print("\nThis image contains multiple boxes (objects):")
    for i, b in enumerate(boxes, start=1):
        print(f"  {i}. BCS/Label: {b['label']} | BBox: {b['bbox']}")

    while True:
        choice = input(f"\nSelect the target box index (1 to {len(boxes)} - or 'q' to quit): ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(boxes):
                return boxes[idx - 1]
        print(f"Invalid input! Please enter a number between 1 and {len(boxes)}.")


def visualize_pair(pair: Dict[str, Path]) -> None:
    """Visualize the selected image and its corresponding annotation side-by-side with the crop.

    Args:
        pair: Dictionary containing image and annotation paths.
    """
    img_path = pair["image_path"]
    ann_path = pair["annotation_path"]

    # 1. Read the image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"Error: Image {img_path.name} could not be read.")
        return

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img_rgb.shape

    # 2. Read the boxes
    boxes = parse_annotation(ann_path)
    if not boxes:
        print(f"No boxes found in {ann_path.name}.")
        return

    # 3. Select a box
    selected_box = select_box(boxes)
    if selected_box is None:
        return

    xmin, ymin, xmax, ymax = selected_box["bbox"]
    label = selected_box["label"]

    # 4. Control coordinates to prevent going out of image dimensions (Clipping)
    xmin = max(0, min(xmin, w_img - 1))
    ymin = max(0, min(ymin, h_img - 1))
    xmax = max(1, min(xmax, w_img))
    ymax = max(1, min(ymax, h_img))

    if xmin >= xmax or ymin >= ymax:
        print(f"Error: Invalid box dimensions: [{xmin}, {ymin}, {xmax}, {ymax}]")
        return

    # 5. Crop in memory (without saving to disk)
    cropped_img = img_rgb[ymin:ymax, xmin:xmax]

    # 6. Draw rectangle and label on the original image
    img_with_box = img_rgb.copy()
    cv2.rectangle(img_with_box, (xmin, ymin), (xmax, ymax), (255, 0, 0), thickness=3)
    cv2.putText(
        img_with_box,
        f"BCS: {label}",
        (xmin, max(25, ymin - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
        cv2.LINE_AA,
    )

    # 7. Display two windows side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img_with_box)
    axes[0].set_title(f"Original: {img_path.name}\nBBox: [{xmin}, {ymin}, {xmax}, {ymax}]")
    axes[0].axis("off")

    axes[1].imshow(cropped_img)
    axes[1].set_title(f"Cropped (BCS: {label})\nShape: {cropped_img.shape[1]}x{cropped_img.shape[0]} (WxH)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main() -> None:
    """Execute the main inspection script."""
    if not DATASET_DIR.exists():
        print(f"Folder '{DATASET_DIR.resolve()}' not found! Make sure you are in the project root.")
        return

    pairs = get_all_pairs()
    if not pairs:
        print("No image and XML/HTML pairs found. Check the dataset folder structure.")
        return

    pair = select_pair(pairs)
    if pair:
        visualize_pair(pair)


if __name__ == "__main__":
    main()