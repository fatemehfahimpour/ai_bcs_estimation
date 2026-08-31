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

# Directory containing the original raw dataset
DATASET_DIR = Path("../dataset")

# Directory containing the train/val/test split metadata CSV files
SPLITS_DIR = Path("../meta_data") / "splits"

# Output directory where the YOLO-formatted dataset will be saved
OUTPUT_DIR = Path("detector_dataset")

# Output YAML configuration file path for Ultralytics YOLO training
DATA_YAML_PATH = Path("data_detector.yaml")

# Supported image file extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# If True, expand the original XML bbox to an isotropic square scaled by SQUARE_SCALE.
# If False, use the original rectangular bbox from the XML annotation.
NOT_USE_XML_BBOX = True
SQUARE_SCALE = 1.5

# Detection class mapping
CLASS_ID = 0
CLASS_NAME = "cow_anatomical_region"

# Seed for reproducibility
RANDOM_SEED = 42

# ============================================================
# File discovery
# ============================================================


def normalize_filename(value: object) -> str:
    """Normalize a raw filename or file path string.

    Cleans whitespace and replaces backward slashes with forward slashes
    to ensure cross-platform compatibility across operating systems.

    Args:
        value (object): Raw filename or path entry from a CSV record.

    Returns:
        str: Normalized, stripped, and forward-slash formatted path string.
    """
    return str(value).strip().replace("\\", "/")


def get_filename_column(df: pd.DataFrame) -> str:
    """Identify the column name corresponding to image filenames in the DataFrame.

    Args:
        df (pd.DataFrame): Input split metadata DataFrame.

    Returns:
        str: Column identifier for image filenames (defaults to 'file_name').
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
    """Locate an image and its corresponding Pascal VOC XML annotation file.

    Performs a direct path resolution first. If not found directly, executes a
    recursive search across the entire `DATASET_DIR` hierarchy.

    Args:
        file_name (str): Relative path or base filename of the requested image.

    Returns:
        tuple[Path, Path]: A tuple containing `(image_path, xml_path)`.

    Raises:
        RuntimeError: If multiple candidate pairs are found for the same filename.
        FileNotFoundError: If no valid image/XML pair is found within the dataset.
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

    # Case 2: CSV contains only the filename, search recursively
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
    """Parse all bounding boxes from a Pascal VOC XML annotation file.

    Extracts bounding boxes defined under each `<object><bndbox>` tag and
    verifies validity (e.g., ensuring `xmax > xmin` and `ymax > ymin`).

    Args:
        xml_path (Path): Path to the Pascal VOC XML file.

    Returns:
        list[tuple[float, float, float, float]]: List of valid bounding boxes
            represented as `(xmin, ymin, xmax, ymax)` in absolute pixel coordinates.

    Raises:
        ValueError: If any coordinate tag (`xmin`, `ymin`, `xmax`, `ymax`) is missing.
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

        # Check for degenerate or inverted boxes
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
    """Clip absolute-pixel bounding box coordinates to image canvas boundaries.

    Args:
        xmin (float): Left coordinate.
        ymin (float): Top coordinate.
        xmax (float): Right coordinate.
        ymax (float): Bottom coordinate.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        tuple[float, float, float, float]: Clipped coordinates
            `(xmin, ymin, xmax, ymax)` within `[0, width]` and `[0, height]`.
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
    """Transform a rectangular bounding box into an expanded isotropic square.

    Calculates the center and the maximum dimension (width or height), applies
    the scale factor, and computes new square coordinates centered at the original box.

    Note:
        Coordinates are unclipped and might lie outside image boundaries.
        Call `clip_bbox` afterwards.

    Args:
        xmin (float): Left coordinate.
        ymin (float): Top coordinate.
        xmax (float): Right coordinate.
        ymax (float): Bottom coordinate.
        scale (float): Expansion scaling factor applied to the maximum dimension.

    Returns:
        tuple[float, float, float, float]: Expanded square bounding box `(xmin, ymin, xmax, ymax)`.
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
    """Convert absolute pixel coordinates to normalized YOLO format.

    Args:
        xmin (float): Left coordinate in pixels.
        ymin (float): Top coordinate in pixels.
        xmax (float): Right coordinate in pixels.
        ymax (float): Bottom coordinate in pixels.
        image_width (int): Width of the image in pixels.
        image_height (int): Height of the image in pixels.

    Returns:
        tuple[float, float, float, float]: Normalized bounding box
            `(x_center, y_center, width, height)` with values scaled to `[0.0, 1.0]`.
    """
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
    """Sanitize a string to create a filesystem-safe filename component.

    Replaces any character that is not alphanumeric, underscore, period,
    or hyphen with an underscore.

    Args:
        value (str): Original string/folder/file name.

    Returns:
        str: Sanitized filesystem-safe string.
    """
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def make_unique_output_stem(image_path: Path) -> str:
    """Construct a unique output file stem incorporating the parent folder name.

    Prevents filename collisions when identical image names exist across
    different subdirectories (e.g., BCS score folders).
    Example: `dataset/3.25/GS_1_2.jpg` -> `bcs_3.25__GS_1_2`.

    Args:
        image_path (Path): Path to the source image file.

    Returns:
        str: Collison-safe unique filename stem.
    """
    parent_name = sanitize_name(image_path.parent.name)
    image_stem = sanitize_name(image_path.stem)
    return f"bcs_{parent_name}__{image_stem}"


# ============================================================
# Dataset generation
# ============================================================


def process_split(split_name: str, dataframe: pd.DataFrame) -> tuple[int, int]:
    """Process a single dataset split (train, val, or test).

    Reads image annotations, formats bounding boxes (optionally expanding to square),
    normalizes coordinates to YOLO format, copies images, and writes label text files.

    Args:
        split_name (str): Split partition identifier ('train', 'val', or 'test').
        dataframe (pd.DataFrame): DataFrame containing records for the split.

    Returns:
        tuple[int, int]: Counts of `(processed_count, skipped_count)` samples.
    """
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

        # Attempt to locate source image and corresponding XML annotation
        try:
            image_path, xml_path = find_image_and_xml(file_name)
        except (FileNotFoundError, RuntimeError) as error:
            print(f"\nWarning: {error}")
            skipped_count += 1
            continue

        # Extract image dimensions
        with Image.open(image_path) as image:
            image_width, image_height = image.size

        # Parse ground-truth boxes from XML
        boxes = parse_voc_objects(xml_path)

        if not boxes:
            print(f"\nWarning: no valid objects found in {xml_path}")
            skipped_count += 1
            continue

        output_stem = make_unique_output_stem(image_path)
        output_image_path = (
            image_output_dir / f"{output_stem}{image_path.suffix.lower()}"
        )
        output_label_path = label_output_dir / f"{output_stem}.txt"

        # Copy image file to the split image directory
        shutil.copy2(image_path, output_image_path)

        label_lines = []

        # Convert and format bounding boxes
        for xmin, ymin, xmax, ymax in boxes:
            if NOT_USE_XML_BBOX:
                xmin, ymin, xmax, ymax = make_square_bbox(
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                    scale=SQUARE_SCALE,
                )

            # Ensure coordinates stay strictly within image boundaries
            xmin, ymin, xmax, ymax = clip_bbox(
                xmin,
                ymin,
                xmax,
                ymax,
                image_width,
                image_height,
            )

            # Ignore zero-area or invalid boxes after clipping
            if xmax <= xmin or ymax <= ymin:
                continue

            # Convert to YOLO format (class_id x_center y_center width height)
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

        # Cleanup if no usable labels survived conversion
        if not label_lines:
            print(f"\nWarning: no usable labels generated for {image_path}")
            output_image_path.unlink(missing_ok=True)
            skipped_count += 1
            continue

        # Save YOLO annotation text file
        output_label_path.write_text(
            "\n".join(label_lines) + "\n",
            encoding="utf-8",
        )

        processed_count += 1

    return processed_count, skipped_count


def write_data_yaml() -> None:
    """Generate the Ultralytics dataset configuration YAML file.

    Writes the dataset root path, paths to split subsets, and class definitions
    required by YOLO training routines.
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
    """Execute the end-to-end dataset conversion pipeline.

    Validates split metadata files, wipes any existing output directory,
    processes all splits, and creates the YAML dataset configuration file.
    """
    split_files = {
        "train": SPLITS_DIR / "train.csv",
        "val": SPLITS_DIR / "val.csv",
        "test": SPLITS_DIR / "test.csv",
    }

    # Verify presence of split metadata files
    for split_name, csv_path in split_files.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")

    # Remove stale dataset directory if already present
    if OUTPUT_DIR.exists():
        print(f"Removing previous generated dataset: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    total_processed = 0
    total_skipped = 0

    # Process each partition
    for split_name, csv_path in split_files.items():
        dataframe = pd.read_csv(csv_path)

        processed, skipped = process_split(split_name, dataframe)
        total_processed += processed
        total_skipped += skipped

        print(f"{split_name}: processed={processed}, skipped={skipped}")

    # Generate data_detector.yaml
    write_data_yaml()

    print("\nDataset preparation completed.")
    print(f"Generated dataset: {OUTPUT_DIR.resolve()}")
    print(f"Generated YAML: {DATA_YAML_PATH.resolve()}")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")


if __name__ == "__main__":
    main()
