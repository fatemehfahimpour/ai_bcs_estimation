from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from data.preprocess import (
    TRAIN_PATH,
    VAL_PATH,
    TEST_PATH,
    get_train_transform,
    get_val_transform,
    CLASS_MAPPING,
    resolve_image_path,
)


class OrdinalBCSDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):

        row = self.dataframe.iloc[index]

        image_path = resolve_image_path(str(row["path"]))

        if image_path is None or not image_path.exists():
            raise FileNotFoundError(
                f"Image not found at: {row['path']}"
            )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        bcs = float(row["bcs"])

        if bcs not in CLASS_MAPPING:
            raise ValueError(
                f"Unknown BCS value {bcs} at index {index}"
            )

        class_index = CLASS_MAPPING[bcs]

        # -----------------------------------------
        # Ordinal target
        # -----------------------------------------

        ordinal_target = torch.zeros(
            4,
            dtype=torch.float32
        )

        if class_index > 0:
            ordinal_target[0] = 1

        if class_index > 1:
            ordinal_target[1] = 1

        if class_index > 2:
            ordinal_target[2] = 1

        if class_index > 3:
            ordinal_target[3] = 1

        return (
            image,
            ordinal_target,
            torch.tensor(class_index, dtype=torch.long)
        )


def get_ordinal_data_loader(
        batch_size=128,
        num_workers=6
):
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # استفاده از همان preprocessing اصلی
    train_dataset = OrdinalBCSDataset(
        train_df,
        transform=get_train_transform()
    )

    val_dataset = OrdinalBCSDataset(
        val_df,
        transform=get_val_transform()
    )

    test_dataset = OrdinalBCSDataset(
        test_df,
        transform=get_val_transform()
    )

    use_cuda = torch.cuda.is_available()

    loader_kwargs = {
        "pin_memory": use_cuda,
        "num_workers": num_workers
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs
    )

    return train_loader, val_loader, test_loader
