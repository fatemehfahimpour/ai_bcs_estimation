"""High-throughput Dataset and DataLoader module for Cow Body Condition Score (BCS) classification.

Provides aspect-ratio preserving letterbox padding pipelines (neutral gray padding),
custom Dataset handling, and accelerated PyTorch DataLoader configurations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================================================
# Paths & Project Structure
# ============================================================
CURRENT_DIR: Path = Path(__file__).resolve().parent

PROJECT_ROOT: Path = CURRENT_DIR.parent

TRAIN_PATH: Path = CURRENT_DIR / "meta_data" / "splits" / "train.csv"
VAL_PATH: Path = CURRENT_DIR / "meta_data" / "splits" / "val.csv"
TEST_PATH: Path = CURRENT_DIR / "meta_data" / "splits" / "test.csv"

PREVIEW_DIR: Path = CURRENT_DIR / "augmentation_preview"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Constants & Hyperparameters
# ============================================================
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

CLASS_MAPPING: Dict[float, int] = {3.25: 0, 3.50: 1, 3.75: 2, 4.00: 3, 4.25: 4}
INDEX_TO_BCS: Dict[int, float] = {v: k for k, v in CLASS_MAPPING.items()}

BATCH_SIZE: int = 128
NUM_CLASSES: int = len(CLASS_MAPPING)
NUM_WORKERS: int = 4 if torch.cuda.is_available() else 0

TARGET_SIZE: int = 224
GRAY_FILL: Tuple[int, int, int] = (128, 128, 128)  # Neutral gray color


def resolve_image_path(raw_path_str: str) -> Optional[Path]:
    """
    Resolve absolute path of an image using relative paths from metadata CSV.

    Args:
        raw_path_str (str): Raw path string from dataset record.

    Returns:
        Optional[Path]: Validated Path object if found, otherwise None.
    """
    p = Path(raw_path_str)
    # 1. Absolute path check
    if p.is_absolute() and p.exists():
        return p
    # 2. Relative to current module directory
    cand1 = (CURRENT_DIR / p).resolve()
    if cand1.exists():
        return cand1
    # 3. Relative to project root
    cand2 = (PROJECT_ROOT / p).resolve()
    if cand2.exists():
        return cand2
    return None


# ============================================================
# Aspect-Ratio Preserving Transforms (Zero-Crop Policy)
# ============================================================
class MakeSquareWithGrayPadding:
    """
    Transform image into a square canvas using symmetric neutral gray padding.

    Preserves original aspect ratio and dimensions without cropping content.
    """

    def __init__(self, fill: Tuple[int, int, int] = GRAY_FILL) -> None:
        """
        Initialize padding transformation.

        Args:
            fill (Tuple[int, int, int]): RGB fill color tuple for padding borders.
        """
        self.fill: Tuple[int, int, int] = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply square padding to input PIL image.

        Args:
            img (Image.Image): Input PIL image.

        Returns:
            Image.Image: Symmetrically padded square PIL image.
        """
        w, h = img.size
        if w == h:
            return img

        max_dim = max(w, h)
        pad_left = (max_dim - w) // 2
        pad_right = max_dim - w - pad_left
        pad_top = (max_dim - h) // 2
        pad_bottom = max_dim - h - pad_top

        return ImageOps.expand(
            img,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=self.fill,
        )


class SafeGrayRotation:
    """
    Apply random rotation expanding canvas bounds and padding margins with gray.

    Guarantees zero-crop policy by expanding image dimensions to fit rotated content.
    """

    def __init__(self, degrees: float = 10.0, fill: Tuple[int, int, int] = GRAY_FILL) -> None:
        """
        Initialize safe rotation transformation.

        Args:
            degrees (float): Maximum rotation angle range [-degrees, degrees].
            fill (Tuple[int, int, int]): RGB fill color tuple for exposed borders.
        """
        self.degrees: float = degrees
        self.fill: Tuple[int, int, int] = fill

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Rotate image within uniform degree range with expanded canvas.

        Args:
            img (Image.Image): Input PIL image.

        Returns:
            Image.Image: Rotated PIL image with preserved corner boundaries.
        """
        angle = float(torch.empty(1).uniform_(-self.degrees, self.degrees).item())
        return img.rotate(
            angle,
            resample=Image.Resampling.BILINEAR,
            expand=True,
            fillcolor=self.fill,
        )


# ============================================================
# Transforms Pipelines
# ============================================================
def get_train_transform(target_size: int = TARGET_SIZE) -> transforms.Compose:
    """
    Build training augmentation pipeline with zero-crop gray letterboxing.

    Args:
        target_size (int): Final square image dimension.

    Returns:
        transforms.Compose: Composed training transformation pipeline.
    """
    return transforms.Compose(
        [
            MakeSquareWithGrayPadding(fill=GRAY_FILL),
            transforms.RandomHorizontalFlip(p=0.5),
            SafeGrayRotation(degrees=10.0, fill=GRAY_FILL),
            MakeSquareWithGrayPadding(fill=GRAY_FILL),
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(
                brightness=0.6,
                contrast=0.6,
                saturation=0.6,
                hue=0.12,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transform(target_size: int = TARGET_SIZE) -> transforms.Compose:
    """
    Build validation preprocessing pipeline with zero-crop gray letterboxing.

    Args:
        target_size (int): Final square image dimension.

    Returns:
        transforms.Compose: Composed validation transformation pipeline.
    """
    return transforms.Compose(
        [
            MakeSquareWithGrayPadding(fill=GRAY_FILL),
            transforms.Resize((target_size, target_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_test_transform(target_size: int = TARGET_SIZE) -> transforms.Compose:
    """
    Build test evaluation pipeline with zero-crop gray letterboxing.

    Args:
        target_size (int): Final square image dimension.

    Returns:
        transforms.Compose: Composed test transformation pipeline.
    """
    return get_val_transform(target_size=target_size)


# ============================================================
# Dataset Class
# ============================================================
class BCSDataset(Dataset):
    """
    PyTorch Dataset wrapper for Cow Body Condition Score (BCS) metadata records.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        """
        Initialize dataset from metadata DataFrame.

        Args:
            dataframe (pd.DataFrame): Metadata DataFrame with image paths and BCS labels.
            transform (Optional[transforms.Compose]): Image transformation pipeline.
        """
        self.dataframe: pd.DataFrame = dataframe.reset_index(drop=True)
        self.transform: Optional[transforms.Compose] = transform

    def __len__(self) -> int:
        """
        Return total sample count in dataset.

        Returns:
            int: Number of records.
        """
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fetch transformed image tensor and class index label for a sample.

        Args:
            index (int): Sample record index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image tensor and class label tensor.
        """
        row = self.dataframe.iloc[index]

        image_path = resolve_image_path(str(row["path"]))
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {row['path']}")

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        bcs = float(row["bcs"])
        if bcs not in CLASS_MAPPING:
            raise ValueError(f"Unknown BCS value {bcs} in row {index}")

        label = torch.tensor(CLASS_MAPPING[bcs], dtype=torch.long)
        return image, label


# ============================================================
# DataLoader Factory
# ============================================================
def get_data_loader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Construct high-throughput train, validation, and test PyTorch DataLoaders.

    Args:
        batch_size (int): Mini-batch size for loaders.
        num_workers (int): Number of background data-loading subprocesses.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test loaders.
    """
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_dataset = BCSDataset(train_df, transform=get_train_transform())
    val_dataset = BCSDataset(val_df, transform=get_val_transform())
    test_dataset = BCSDataset(test_df, transform=get_test_transform())

    use_cuda = torch.cuda.is_available()
    loader_kwargs: Dict[str, Any] = {
        "pin_memory": use_cuda,
        "num_workers": num_workers,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# ============================================================
# Preview & Visualization Utility
# ============================================================
def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """
    Denormalize ImageNet-scaled PyTorch tensor back to [0, 1] RGB NumPy array.

    Args:
        tensor (torch.Tensor): Normalized input tensor of shape (3, H, W).

    Returns:
        np.ndarray: Denormalized image array of shape (H, W, 3).
    """
    tensor_copy = tensor.clone().detach().cpu()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor_copy = tensor_copy * std + mean
    tensor_copy = torch.clamp(tensor_copy, 0.0, 1.0)
    return tensor_copy.permute(1, 2, 0).numpy()


def save_augmentation_preview(
    image_path: Optional[Union[str, Path]] = None,
    num_samples: int = 4,
    output_dir: Path = PREVIEW_DIR,
) -> None:
    """
    Generate and save step-by-step visual comparison of augmentation pipeline.

    Args:
        image_path (Optional[Union[str, Path]]): Specific input image path.
        num_samples (int): Number of augmented variants to render.
        output_dir (Path): Destination directory for generated preview images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    src_path: Optional[Path] = None
    bcs_val: str = "Unknown"

    if image_path is not None:
        p = resolve_image_path(str(image_path))
        if p and p.exists():
            src_path = p
            bcs_val = "Manual"

    if src_path is None and TRAIN_PATH.exists():
        train_df = pd.read_csv(TRAIN_PATH)
        for _, row in train_df.iterrows():
            cand = resolve_image_path(str(row["path"]))
            if cand and cand.exists():
                src_path = cand
                bcs_val = str(row["bcs"])
                break

    if src_path is None:
        for root_check in [PROJECT_ROOT, CURRENT_DIR]:
            cropped_dir = root_check / "cropped_dataset"
            imgs = list(cropped_dir.glob("**/*.jpg")) + list(cropped_dir.glob("**/*.png"))
            if imgs:
                src_path = imgs[0]
                bcs_val = "Discovered"
                break

    if src_path is None:
        print("[!] No image found to preview.")
        print(f"    Checking TRAIN_PATH at: {TRAIN_PATH}")
        return

    print(f"[*] Processing image: {src_path}")
    orig_img = Image.open(src_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    # Step 1: Gray padding to match shorter dimension to larger dimension
    square_transform = MakeSquareWithGrayPadding(fill=GRAY_FILL)
    square_img = square_transform(orig_img)
    sq_w, sq_h = square_img.size

    # PIL pilot pipeline to visualize rotation and padding before tensor normalization
    aug_pilot = transforms.Compose(
        [
            MakeSquareWithGrayPadding(fill=GRAY_FILL),
            transforms.RandomHorizontalFlip(p=0.5),
            SafeGrayRotation(degrees=10.0, fill=GRAY_FILL),
            MakeSquareWithGrayPadding(fill=GRAY_FILL),
            transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        ]
    )

    full_train_transform = get_train_transform(target_size=TARGET_SIZE)

    # Save intermediate image artifacts
    orig_img.save(output_dir / "00_original_crop.jpg")
    square_img.save(output_dir / "01_padded_to_square_raw.jpg")

    aug_images = []
    tensor_views = []
    for i in range(num_samples):
        aug_p = aug_pilot(orig_img)
        aug_p.save(output_dir / f"02_augmented_sample_{i+1}.jpg")
        aug_images.append(aug_p)

        t_img = full_train_transform(orig_img)
        tensor_views.append(t_img)

    # Visual grid comparison
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Strict Zero-Crop Pipeline (Pure Gray Padded) | BCS: {bcs_val}\n"
        f"Original: {orig_w}x{orig_h} -> Padded Square: {sq_w}x{sq_h} -> Model Input: {TARGET_SIZE}x{TARGET_SIZE}",
        fontsize=12,
        fontweight="bold",
    )

    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title(f"1. Original Crop\n({orig_w}x{orig_h})", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(square_img)
    axes[0, 1].set_title(f"2. Gray Padded Square\n({sq_w}x{sq_h})", fontsize=10)
    axes[0, 1].axis("off")

    for idx, (ax, aug) in enumerate(
        [
            (axes[0, 2], aug_images[0]),
            (axes[0, 3], aug_images[1]),
            (axes[1, 0], aug_images[2]),
            (axes[1, 1], aug_images[3]),
        ]
    ):
        ax.imshow(aug)
        ax.set_title(f"Augmentation #{idx+1}\n(Rotated & Gray Padded)", fontsize=10)
        ax.axis("off")

    denorm_img = denormalize_tensor(tensor_views[0])
    axes[1, 2].imshow(denorm_img)
    axes[1, 2].set_title("De-normalized Tensor\n(224x224)", fontsize=10)
    axes[1, 2].axis("off")

    norm_vis = tensor_views[0].permute(1, 2, 0).numpy()
    norm_vis = (norm_vis - norm_vis.min()) / (norm_vis.max() - norm_vis.min() + 1e-8)
    axes[1, 3].imshow(norm_vis)
    axes[1, 3].set_title("Exact ResNet-18 Input", fontsize=10)
    axes[1, 3].axis("off")

    plt.tight_layout()
    overview_path = output_dir / "pipeline_overview.png"
    plt.savefig(overview_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[✔] Augmentation preview successfully saved to {output_dir.resolve()}")


if __name__ == "__main__":
    if __name__ == "__main__":
        # Set absolute or relative path to target test image:
        my_image = r"E:\ai_bcs_estimation\cropped_dataset\4.0\L-i9987.jpg"
        save_augmentation_preview(image_path=my_image, num_samples=4)
