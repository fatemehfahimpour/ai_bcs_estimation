import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASS_MAPPING = {3.25: 0, 3.50: 1, 3.75: 2, 4.00: 3, 4.25: 4}
INDEX_TO_BCS = {0: 3.25, 1: 3.50, 2: 3.75, 3: 4.00, 4: 4.25}  # Reverse mapping

BATCH_SIZE = 32
NUM_CLASSES = len(CLASS_MAPPING)
NUM_WORKERS = 0

TRAIN_PATH = 'meta_data/splits/train.csv'
VAL_PATH = 'meta_data/splits/val.csv'
TEST_PATH = 'meta_data/splits/test.csv'


def resize():
    # خانم غضنفری لطفا این بخش تکمیل گردد
    # با تشکر از زحمات فراوان شما
    # ارادتمند فهیم پور
    # امضا و اثر انگشت
    # 😘😘😘😘😘😘😘😘
    return transforms.Resize((224, 224))


def get_train_transform():
    train_transform = transforms.Compose([
        resize(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=10
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        # Convert image to Tensor
        transforms.ToTensor(),
        # ImageNet normalization
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])
    return train_transform


def get_val_transform():
    val_transform = transforms.Compose([
        resize(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])

    return val_transform


def get_test_transform():
    test_transform = transforms.Compose([
        resize(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])
    return test_transform


class BCSDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = row['path']
        image = Image.open(image_path)

        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        bcs = float(row['bcs'])

        if bcs not in CLASS_MAPPING:
            raise ValueError(f'Unknown BCS value: {bcs}')

        label = CLASS_MAPPING[bcs]
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_data_loader(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)

    train_transform = get_train_transform()
    val_transform = get_val_transform()
    test_transform = get_test_transform()

    train_dataset = BCSDataset(train_df, transform=train_transform)
    val_dataset = BCSDataset(val_df, transform=val_transform)
    test_dataset = BCSDataset(test_df, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return (train_loader, val_loader, test_loader)


if __name__ == '__main__':
    # for test
    train_loader, val_loader, test_loader = get_data_loader()
    images, labels = next(iter(train_loader))

    print('Images shape:', images.shape)
    print('Labels shape:', labels.shape)
    print('Labels:', labels)
    print('Labels dtype:', labels.dtype)
