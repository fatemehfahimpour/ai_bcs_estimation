from __future__ import annotations

import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("dataset")
SPLITS_DIR = Path("meta_data") / "splits"
OUTPUT_DIR = Path("detector_dataset")
DATA_YAML_PATH = Path("data_detector.yaml")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# اگر True باشد، bbox XML به مربع 1.5 برابری تبدیل می‌شود.
# اگر False باشد، bbox مستطیلی اصلی XML استفاده خواهد شد.
NOT_USE_XML_BBOX = True
SQUARE_SCALE = 1.5

CLASS_ID = 0
CLASS_NAME = "cow_anatomical_region"

RANDOM_SEED = 42

# ============================================================
# File discovery
# ============================================================

def normalize_filename(value: object) -> str:
    """Return a normalized filename string from a CSV value"""
    return str(value).strip().replace("\\", "/")

def get_filename_column(df: pd.DataFrame) -> str:
    """
    Fing the image filename column used bu all_images.csv and split CSVs.
    """
    # candidates = [
    #     "file_name",
    #     "filename",
    #     "image_name",
    #     "image",
    #     "img_name",
    # ]

    # for column in candidates:
    #     if column in df.columns:
    #         return column

    # raise ValueError(
    #     "No image filename column was found. "
    #     f"Available columns: {list(df.columns)}"
    # )
    return "file_name"

def find_image_and_xml(file_name: str) -> tuple[Path, Path]:
    """
    Fing an image and its sibling Pascal VOC XML anywhere below dataset/.
    """
    normalized = normalize_filename(file_name)
    requested_path = Path(normalized)

    # Case 1: CSV contains a path relative to project root or dataset/
    direct_candidates = [
        Path(normalized),
        DATASET_DIR / normalized,
    ]

    for image_path in direct_candidates:
        if image_path.exists() and image_path.suffix in IMAGE_EXTENSIONS:
            xml_path = image_path.with_suffix(".xml")
            if xml_path.exists():
                return image_path, xml_path

    # Case 2: CSV contains only the filename
    image_name = requested_path.name
    matches = [
        path
        for path in DATASET_DIR.rglob(image_name)
        if path.is_file() and path.suffix in IMAGE_EXTENSIONS
    ]

    valid_pairs = []
    for image_path in matches:
        xml_path = image_path.with_suffix(".xml")
        if xml_path.exists():
            valid_pairs.append((image_path, xml_path))

    if len(valid_pairs) == 1:
        return valid_pairs[0]

    if len(valid_pairs) > 1:
        paths = "\n".join(str(pair[0]) for pair in valid_pairs)
        raise RuntimeError(
            f"Multiple image/XML pairs found for '{file_name}':\n{paths}\n"
            "The CSV must contain a unique filename or relative path."
        )

    raise FileNotFoundError(
        f"Could not find image/XML pair for '{file_name}' below '{DATASET_DIR}'."
    )

# ============================================================
# XML and bounding box processing
# ============================================================

def parse_voc_objects(xml_path: Path) -> list[tuple[float, float, float, float]]:
    """
    Read all Pascal VOC bounding boxes from an XML file
    """  
    root = ET.parse(xml_path).getroot()
    boxes = []

    for object_node in root.findall("object"):
        bbox_node = object_node.find("bndbox")
        if bbox_node is None:
            continue

        values = []
        for tag in ("xmin", "ymin", "xmax", "ymax"):
            value = bbox_node.findtext(tag)
            if value is None:
                raise ValueError(f"Missing '{tag}' in XML: {xml_path}")
            values.append(float(value))

        xmin, ymin, xmax, ymax = values

        if xmax <= xmin or ymax <= ymin:
            print(f"Warning: invalid bbox in {xml_path}: {values}")
            continue

        boxes.append((xmin, ymin, xmax, ymax))

    return boxes


def clip_bbox(
        xmin: float,
        ymin: float,
        xmax: float,
        ymax: float,
        image_width: int,
        image_height: int,
) -> tuple[float, float, float, float]:
    """
    Clip an absolute-pixel bbox to image boundaries.
    """
    xmin = max(0.0, min(float(image_width), xmin))
    ymin = max(0.0, min(float(image_height), ymin))
    xmax = max(0.0, min(float(image_width), xmax))
    ymax = max(0.0, min(float(image_height), ymax))

    return xmin, ymin, xmax, ymax

def make_square_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    scale: float,
) -> tuple[float, float, float, float]:
    """
    Create a square around the original bbox center.

    The square is not clipped here. It may extend beyond the image.
    It will be clipped later because YOLO labels cannot contain coordinates
    outside the image.
    """
    width = xmax - xmin
    height = ymax - ymin

    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    side = scale * max(width, height)

    return (
        center_x - side / 2.0,
        center_y - side / 2.0,
        center_x + side / 2.0,
        center_y + side / 2.0,
    )

def absolute_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert an absolute-pixel bbox to normalized YOLO format."""
    width = xmax - xmin
    height = ymax - ymin

    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    return (
        center_x / image_width,
        center_y / image_height,
        width / image_width,
        height / image_height,
    )

def sanitize_name(value: str) -> str:
    """Make a safe filename component."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

def make_unique_output_stem(image_path: Path) -> str:
    """
    Avoid filename collisions between BCS folders.
    Example: dataset/3.25/GS_1_2.jpg -> bcs_3.25__GS_1_2
    """
    parent_name = sanitize_name(image_path.parent.name)
    image_stem = sanitize_name(image_path.stem)
    return f"bcs_{parent_name}__{image_stem}"

# ============================================================
# Dataset generation
# ============================================================

def process_split(split_name: str, dataframe: pd.DataFrame) -> tuple[int, int]:
    image_output_dir = OUTPUT_DIR / "images" / split_name
    label_output_dir = OUTPUT_DIR / "labels" / split_name

    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    filename_column = get_filename_column(dataframe)

    processed_count = 0
    skipped_count = 0

    for _, row in tqdm(
        dataframe.iterrows(),
        total=len(dataframe),
        desc=f"Preparing {split_name}",
    ):
        file_name = normalize_filename(row[filename_column])

        try:
            image_path, xml_path = find_image_and_xml(file_name)
        except (FileNotFoundError, RuntimeError) as error:
            print(f"\nWarning: {error}")
            skipped_count += 1
            continue

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        boxes = parse_voc_objects(xml_path)

        if not boxes:
            print(f"\nWarning: no valid objects found in {xml_path}")
            skipped_count += 1
            continue

        output_stem = make_unique_output_stem(image_path)
        output_image_path = image_output_dir / f"{output_stem}{image_path.suffix.lower()}"
        output_label_path = label_output_dir / f"{output_stem}.txt"

        shutil.copy2(image_path, output_image_path)

        label_lines = []

        for xmin, ymin, xmax, ymax in boxes:
            if NOT_USE_XML_BBOX:
                xmin, ymin, xmax, ymax = make_square_bbox(
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    scale=SQUARE_SCALE,
                )

            # A target extending outside the image must be clipped for YOLO.
            xmin, ymin, xmax, ymax = clip_bbox(
                xmin,
                ymin,
                xmax,
                ymax,
                image_width,
                image_height,
            )

            if xmax <= xmin or ymax <= ymin:
                continue

            x_center, y_center, width, height = absolute_to_yolo(
                xmin,
                ymin,
                xmax,
                ymax,
                image_width,
                image_height,
            )

            label_lines.append(
                f"{CLASS_ID} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

        if not label_lines:
            print(f"\nWarning: no usable labels generated for {image_path}")
            output_image_path.unlink(missing_ok=True)
            skipped_count += 1
            continue

        output_label_path.write_text(
            "\n".join(label_lines) + "\n",
            encoding="utf-8",
        )

        processed_count += 1

    return processed_count, skipped_count

def write_data_yaml() -> None:
    """
    Create the Ultralytics data configuration.
    """
    absolute_dataset_path = OUTPUT_DIR.resolve().as_posix()

    yaml_content = (
        f"path: '{absolute_dataset_path}'\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )

    DATA_YAML_PATH.write_text(yaml_content, encoding="utf-8")

def main() -> None:
    split_files = {
        "train": SPLITS_DIR / "train.csv",
        "val": SPLITS_DIR / "val.csv",
        "test": SPLITS_DIR / "test.csv",
    }

    for split_name, csv_path in split_files.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")

    if OUTPUT_DIR.exists():
        print(f"Removing previous generated dataset: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    total_processed = 0
    total_skipped = 0

    for split_name, csv_path in split_files.items():
        dataframe = pd.read_csv(csv_path)

        processed, skipped = process_split(split_name, dataframe)
        total_processed += processed
        total_skipped += skipped

        print(
            f"{split_name}: processed={processed}, skipped={skipped}"
        )

    write_data_yaml()

    print("\nDataset preparation completed.")
    print(f"Generated dataset: {OUTPUT_DIR.resolve()}")
    print(f"Generated YAML: {DATA_YAML_PATH.resolve()}")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")


if __name__ == "__main__":
    main()

        