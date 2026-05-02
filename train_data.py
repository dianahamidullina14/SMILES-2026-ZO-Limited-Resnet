import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from augmentation import get_transforms


class CIFAR100Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        with open(data_path, "rb") as f:
            d = pickle.load(f, encoding="bytes")
        self.images = d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.labels = d[b"fine_labels"]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.fromarray(self.images[idx])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


def get_train_dataset_loader(data_dir, batch_size, generator_train=None):
    transform = get_transforms(train=True)
    dataset = CIFAR100Dataset(
        "/kaggle/input/datasets/fedesoriano/cifar100/train",
        transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        generator=generator_train,
    )
    return dataset, loader
