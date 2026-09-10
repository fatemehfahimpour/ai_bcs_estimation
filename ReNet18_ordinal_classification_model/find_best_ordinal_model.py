import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_CSV = "../meta_data/splits/train.csv"
VAL_CSV = "../meta_data/splits/val.csv"


