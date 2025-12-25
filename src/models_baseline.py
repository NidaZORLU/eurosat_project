import torch.nn as nn

from .config import NUM_CLASSES


class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        
        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   

            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  

       
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   
        )

      
        self.classifier = nn.Sequential(
            nn.Flatten(),                     
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),               
            nn.Linear(256, NUM_CLASSES),     
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
