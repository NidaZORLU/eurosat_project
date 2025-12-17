
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Tuple

from .config import BATCH_SIZE, VAL_SPLIT, TEST_SPLIT, RANDOM_SEED, IMAGE_SIZE


class EuroSATSubset(Dataset):
    """
    HuggingFace EuroSAT datasetinin belirli index aralığını temsil eden
    PyTorch Dataset wrapperi.
    """
    def __init__(self, hf_dataset, indices, transform=None):
        self.hf_dataset = hf_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        sample = self.hf_dataset[real_idx]

        image = sample["image"]   # PIL Image veya np.array
        label = sample["label"]   # int (0–9)

        # Bazı durumlarda image np.array gelebilir, garanti için PIL'e çevirmmiz gerekiyor
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """
    Train,validation,test için kullanılacak transform pipelineları.
    Trainde augmentation var, val ve testte yok.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
    ])

    # val ve test aynı transform
    return train_transform, eval_transform, eval_transform


def get_dataloaders():
    """
    EuroSAT_RGB datasetini HuggingFace üzerinden indirir,
    stratified şekilde train,val,test split yapar
    ve PyTorch DataLoader objeleri döner.
    """
    print("✅ EuroSAT_RGB dataset indiriliyor / yükleniyor...")
    # split="train": bu dataset 27.000 örneği tek splitte veriyor
    hf_ds = load_dataset("blanchon/EuroSAT_RGB", split="train")

    n_samples = len(hf_ds)
    print(f"Toplam örnek sayısı: {n_samples}")

    indices = np.arange(n_samples)
    labels = np.array(hf_ds["label"])

    # Toplamdan VAL_SPLIT + TEST_SPLIT kadarını val+test için ayıracağız
    test_size = TEST_SPLIT
    val_size = VAL_SPLIT
    temp_size = val_size + test_size

    # 1) Train ile temp (val+test) ayrımı
    train_indices, temp_indices, y_train, y_temp = train_test_split(
        indices,
        labels,
        test_size=temp_size,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    # 2) tempi val ve test diye böleceğiz
    val_rel = val_size / temp_size  # temp içindeki val oranı

    val_indices, test_indices, y_val, y_test = train_test_split(
        temp_indices,
        y_temp,
        test_size=(1 - val_rel),
        random_state=RANDOM_SEED,
        stratify=y_temp,
    )

    print(f"Train size: {len(train_indices)}")
    print(f"Val   size: {len(val_indices)}")
    print(f"Test  size: {len(test_indices)}")

    train_transform, val_transform, test_transform = get_transforms()

    train_dataset = EuroSATSubset(hf_ds, train_indices, transform=train_transform)
    val_dataset = EuroSATSubset(hf_ds, val_indices, transform=val_transform)
    test_dataset = EuroSATSubset(hf_ds, test_indices, transform=test_transform)

    # MacOS + M1 üzerinde stabil çalışsın diye num_workers=0,
    # pin_memory=False (MPS ve CPU'da desteklenmediği için uyarıyı kaldırıyoruz).
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
