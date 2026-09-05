"""High-throughput Dataset and DataLoader module for Cow Body Condition Score (BCS) classification.

Provides aspect-ratio preserving preprocessing pipelines, custom Dataset handling,
and accelerated PyTorch DataLoader configurations optimized for high-throughput GPU training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ============================================================
# Paths & Project Structure
# ============================================================
CURRENT_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = (
    CURRENT_DIR.parent if CURRENT_DIR.name != "ai_bcs_estimation" else CURRENT_DIR
)

TRAIN_PATH: Path = PROJECT_ROOT / "meta_data" / "splits" / "train.csv"
VAL_PATH: Path = PROJECT_ROOT / "meta_data" / "splits" / "val.csv"
TEST_PATH: Path = PROJECT_ROOT / "meta_data" / "splits" / "test.csv"
PREVIEW_DIR: Path = PROJECT_ROOT / "augmentation_preview"

# ============================================================
# Constants & Hyperparameters (Optimized for High-Memory GPUs)
# ============================================================
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

CLASS_MAPPING: Dict[float, int] = {3.25: 0, 3.50: 1, 3.75: 2, 4.00: 3, 4.25: 4}
INDEX_TO_BCS: Dict[int, float] = {v: k for k, v in CLASS_MAPPING.items()}

BATCH_SIZE: int = 64
NUM_CLASSES: int = len(CLASS_MAPPING)
NUM_WORKERS: int = 4 if torch.cuda.is_available() else 0

RESIZE_SIZE: int = 256
CROP_SIZE: int = 224


# ============================================================
# Transforms Pipeline
# ============================================================
def get_train_transform(
    resize_size: int = RESIZE_SIZE, crop_size: int = CROP_SIZE
) -> transforms.Compose:
    """Build the training augmentation pipeline preserving image aspect ratio."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_val_transform(
    resize_size: int = RESIZE_SIZE, crop_size: int = CROP_SIZE
) -> transforms.Compose:
    """Build the deterministic preprocessing pipeline for validation."""
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_test_transform(
    resize_size: int = RESIZE_SIZE, crop_size: int = CROP_SIZE
) -> transforms.Compose:
    """Build the deterministic preprocessing pipeline for test evaluation."""
    return get_val_transform(resize_size=resize_size, crop_size=crop_size)


# ============================================================
# Dataset Class
# ============================================================
class BCSDataset(Dataset):
    """PyTorch Dataset loading cow images and categorical BCS targets from metadata."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.dataframe: pd.DataFrame = dataframe.reset_index(drop=True)
        self.transform: Optional[transforms.Compose] = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]

        raw_path = Path(str(row["path"]))
        image_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at: {image_path}")

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
# High-Throughput DataLoader Factory
# ============================================================
def get_data_loader(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Preview & Visualization Utility
# ============================================================
def denormalize_tensor(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization for displaying PyTorch image tensors."""
    tensor_copy = tensor.clone().detach().cpu()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor_copy = tensor_copy * std + mean
    tensor_copy = torch.clamp(tensor_copy, 0.0, 1.0)
    return tensor_copy.permute(1, 2, 0).numpy()


def save_augmentation_preview(
    image_path: Optional[Path] = None,   # <-- آدرس دستی دلخواه
    num_samples: int = 5,
    output_dir: Path = PREVIEW_DIR,
) -> None:
    """Save step-by-step visual comparisons of transforms.

    Args:
        image_path: Optional manual path to a specific image. If None, uses the
            first sample from train.csv.
        num_samples: Number of random augmentation variants to generate.
        output_dir: Directory to store preview PNG/JPG files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resolve source image --------------------------------------------
    if image_path is not None:
        test_path = Path(image_path)
        if not test_path.exists():
            print(f"[!] Image path does not exist: {test_path}")
            return
        src_path = test_path
        bcs_val = "N/A (Manual Path)"
        print(f"[*] Using manually specified image: {src_path}")
    else:
        if not TRAIN_PATH.exists():
            print(f"[!] {TRAIN_PATH} not found. Skipping preview.")
            return
        train_df = pd.read_csv(TRAIN_PATH)
        if train_df.empty:
            print("[!] train.csv is empty. Skipping preview.")
            return
        sample_row = train_df.iloc[0]
        raw_path = Path(str(sample_row["path"]))
        src_path = raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path
        bcs_val = sample_row["bcs"]
        if not src_path.exists():
            print(f"[!] Sample image not found at {src_path}")
            return

    orig_img = Image.open(src_path).convert("RGB")

    # ---- Base preprocessing: Resize + CenterCrop --------------------------
    base_transform = transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
        ]
    )
    base_img = base_transform(orig_img)

    # ---- Visual (RGB) augmentation pipeline -------------------------------
    aug_pilot = transforms.Compose(
        [
            transforms.Resize(RESIZE_SIZE),
            transforms.CenterCrop(CROP_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03
            ),
        ]
    )

    # ---- Full training transform (the tensor that ResNet actually sees) ---
    full_train_transform = get_train_transform()

    # Save individual source files
    orig_img.save(output_dir / "00_original_cropped.jpg")
    base_img.save(output_dir / "01_resize_center_crop_224.jpg")

    aug_images = []
    tensor_views = []
    for i in range(num_samples):
        aug_p = aug_pilot(orig_img)
        aug_p.save(output_dir / f"02_augmented_sample_{i+1}.jpg")
        aug_images.append(aug_p)

        t_img = full_train_transform(orig_img)
        tensor_views.append(t_img)

    # ---- Build comprehensive overview grid ---------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        f"Input Pipeline Inspection  |  BCS: {bcs_val}\n"
        f"Source: {src_path.name}  ->  Ready for ResNet-18",
        fontsize=13,
        fontweight="bold",
    )

    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title(f"Original Image\n{orig_img.size}", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(base_img)
    axes[0, 1].set_title(f"Base Resize+Crop\n({CROP_SIZE}x{CROP_SIZE})", fontsize=10)
    axes[0, 1].axis("off")

    for idx, (ax, aug) in enumerate(
        [(axes[0, 2], aug_images[0]), (axes[0, 3], aug_images[1]),
         (axes[1, 0], aug_images[2]), (axes[1, 1], aug_images[3])]
    ):
        ax.imshow(aug)
        ax.set_title(f"Augmentation #{idx+1}", fontsize=10)
        ax.axis("off")

    denorm_img = denormalize_tensor(tensor_views[0])
    axes[1, 2].imshow(denorm_img)
    axes[1, 2].set_title("Batch Tensor View\n(De-normalized)", fontsize=10)
    axes[1, 2].axis("off")

    norm_vis = tensor_views[0].permute(1, 2, 0).numpy()
    norm_vis = (norm_vis - norm_vis.min()) / (norm_vis.max() - norm_vis.min())
    axes[1, 3].imshow(norm_vis)
    axes[1, 3].set_title("Exact ResNet-18 Input\n(Normalized Tensor)", fontsize=10)
    axes[1, 3].axis("off")

    plt.tight_layout()
    overview_path = output_dir / "pipeline_overview.png"
    plt.savefig(overview_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"\n[+] Augmentation previews saved to: {output_dir}")
    print(f"[+] Overview image: {overview_path.name}")


# ============================================================
# Main Execution Block
# ============================================================
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_data_loader(batch_size=8, num_workers=0)
    images, labels = next(iter(train_loader))

    print("--- Sanity Check ---")
    print("Device CUDA status:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("Device name:", torch.cuda.get_device_name(0))
    print("Batch images shape:", images.shape)
    print("Batch labels shape:", labels.shape)
    print("Labels tensor:", labels)
    print("BCS Score equivalents:", [INDEX_TO_BCS[int(lbl)] for lbl in labels])

    # ---- Preview with default first image from train.csv ----
    save_augmentation_preview(num_samples=4)

    # ---- Example: manual custom path (uncomment to use) ----
    # save_augmentation_preview(
    #     image_path=PROJECT_ROOT / "dataset" / "custom_folder" / "my_cow.jpg",
    #     num_samples=4
    # )
