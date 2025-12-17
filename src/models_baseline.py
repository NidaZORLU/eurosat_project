import torch.nn as nn

from .config import NUM_CLASSES


class SimpleCNN(nn.Module):
    """
    EuroSAT için basit bir CNN modeli.
    - 3 adet Conv + BatchNorm + ReLU + MaxPool bloğu
    - Fully Connected + Dropout + Output layer
      * Batch Normalization
      * Dropout
      * L2 (weight decay ile) bu kısımda
    """
    def __init__(self):
        super().__init__()

        # Özellik çıkarıcı kısım (convolutional bloklar)
        self.features = nn.Sequential(
            # Blok 1: 3 -> 32 kanal
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 64x64 -> 32x32

            # Blok 2: 32 -> 64 kanal
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 32x32 -> 16x16

            # Blok 3: 64 -> 128 kanal
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # 16x16 -> 8x8
        )

        # Sınıflandırıcı kısım (FC + Dropout)
        self.classifier = nn.Sequential(
            nn.Flatten(),                     # 128 * 8 * 8 = 8192
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),               # Dropout (overfitting'e karşı)
            nn.Linear(256, NUM_CLASSES),     # Çıkış layer: 10 sınıf
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
