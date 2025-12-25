from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple

from .config import BATCH_SIZE, IMAGE_SIZE


class EuroSATSubset(Dataset):

    def __init__(self, hf_dataset, indices, transform=None):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        sample = self.hf_dataset[real_idx]

        image = sample["image"]   
        label = sample["label"]  

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """
    Train, validation, test için kullanılacak transform pipeline'ları.
    Train'de augmentation var, val ve test'te yok.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    return train_transform, eval_transform, eval_transform


def get_dataloaders():
    """
    EuroSAT RGB datasetini HuggingFace üzerinden indirir.
    timm/eurosat-rgb dataset'i hazır train/validation/test splitleriyle gelir:
      train: 16200
      validation: 5400
      test: 5400
    Toplam: 27000
    """
    print("✅ EuroSAT RGB dataset indiriliyor / yükleniyor...")
    ds = load_dataset("timm/eurosat-rgb")  

    train_hf = ds["train"]
    val_hf = ds["validation"]
    test_hf = ds["test"]

    total = len(train_hf) + len(val_hf) + len(test_hf)
    print(f"Toplam örnek sayısı: {total}")
    print(f"Train size: {len(train_hf)}")
    print(f"Val   size: {len(val_hf)}")
    print(f"Test  size: {len(test_hf)}")

    train_transform, val_transform, test_transform = get_transforms()

    train_indices = np.arange(len(train_hf))
    val_indices = np.arange(len(val_hf))
    test_indices = np.arange(len(test_hf))

    train_dataset = EuroSATSubset(train_hf, train_indices, transform=train_transform)
    val_dataset = EuroSATSubset(val_hf, val_indices, transform=val_transform)
    test_dataset = EuroSATSubset(test_hf, test_indices, transform=test_transform)

    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader
