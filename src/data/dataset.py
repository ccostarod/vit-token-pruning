from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102

from data.transforms import get_train_transforms, get_eval_transforms

NUM_CLASSES = 102

def get_train_val_datasets(data_dir: str, image_size: int = 224, download: bool = True):
    data_dir = Path(data_dir)

    train_transform = get_train_transforms(image_size)
    eval_transform = get_eval_transforms(image_size)

    train_dataset = Flowers102(
        root=data_dir,
        split="train",
        transform=train_transform,
        download=download,
    )

    val_dataset = Flowers102(
        root=data_dir,
        split="val",
        transform=eval_transform,
        download=download,
    )

    return train_dataset, val_dataset

def get_test_dataset(data_dir: str, image_size: int = 224, download: bool = True):
    data_dir = Path(data_dir)

    eval_transform = get_eval_transforms(image_size)

    test_dataset = Flowers102(
        root=data_dir,
        split="test",
        transform=eval_transform,
        download=download,
    )

    return test_dataset

def get_train_val_dataloaders(train_dataset, val_dataset, batch_size: int = 32, num_workers: int = 4):

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader

def get_test_dataloader(test_dataset, batch_size: int = 32, num_workers: int = 4):

    pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader



    
