import torch
from PIL import Image
from torchvision import transforms
import sys

from .models_baseline import SimpleCNN
from .config import DEVICE, IMAGE_SIZE

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

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def predict_topk(img_path, k=3):
    image = Image.open(img_path).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(torch.load("baseline_improved_best.pth", map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        outputs = model(image.to(DEVICE))
        probs = torch.softmax(outputs, dim=1)
        topk_probs, topk_idx = torch.topk(probs, k, dim=1)

    topk_probs = topk_probs.cpu().numpy()[0]
    topk_idx = topk_idx.cpu().numpy()[0]

    results = []
    for p, idx in zip(topk_probs, topk_idx):
        results.append((CLASSES[idx], float(p)))
    return results


if __name__ == "__main__":
    img_path = sys.argv[1]
    results = predict_topk(img_path, k=3)

    print("\n🔮 Top-3 Tahmin:")
    for cls, p in results:
        print(f"  {cls:<25} -> {p*100:5.2f}%")
    print()

