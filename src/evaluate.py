import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from .models_baseline import SimpleCNN
from .dataset_eurosat import get_dataloaders
from .config import DEVICE


CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

MODEL_PATH = "baseline_improved_best.pth"  


def evaluate(model_path: str = MODEL_PATH):
    os.makedirs("figures", exist_ok=True)


    train_loader, val_loader, test_loader = get_dataloaders()


    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

 
    report = classification_report(
        y_true, y_pred, target_names=CLASSES, digits=4
    )
    print("\n=== Classification Report (Improved Model) ===")
    print(report)


    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix – Improved Model")
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45, ha="right")
    plt.yticks(tick_marks, CLASSES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("figures/confusion_matrix.png", dpi=200)
    plt.show()

    print("\n✅ Saved: figures/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()

