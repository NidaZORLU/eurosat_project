import torch

from datasets import load_dataset

from .config import DEVICE
from .dataset_eurosat import get_dataloaders
from .models_baseline import SimpleCNN


def load_label_names():
    """
    EuroSAT label isimlerini HuggingFace metadata'sından çeker.
    Sadece feature bilgisi için küçük bir split yeterli.
    """
    ds_info = load_dataset("blanchon/EuroSAT_RGB", split="train[:1]")
    label_names = ds_info.features["label"].names
    return label_names


def load_trained_model():
    """
    Önce improved modeli dene, yoksa baseline'a düş.
    """
    model = SimpleCNN().to(DEVICE)

    tried_paths = ["baseline_improved_best.pth", "baseline_best.pth"]
    loaded = False
    for path in tried_paths:
        try:
            state_dict = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"✅ Model yüklendi: {path}")
            loaded = True
            break
        except FileNotFoundError:
            continue

    if not loaded:
        raise FileNotFoundError(
            "Hiçbir model dosyası bulunamadı. Önce train_baseline veya train_improved çalıştır."
        )

    model.eval()
    return model


def demo_on_test_samples(num_samples: int = 10):
    label_names = load_label_names()
    model = load_trained_model()

    _, _, test_loader = get_dataloaders()

    # Test loader'dan bir batch al
    images, labels = next(iter(test_loader))

    images = images.to(DEVICE)
    labels = labels.to(DEVICE)

    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

    print("\n🧪 Test setinden örnek tahminler:")
    print("-" * 60)
    print(f"{'Idx':<4} {'Gerçek':<20} {'Tahmin':<20} {'Durum':<10}")
    print("-" * 60)

    num = min(num_samples, labels.size(0))

    for i in range(num):
        true_idx = labels[i].item()
        pred_idx = preds[i].item()

        true_name = label_names[true_idx]
        pred_name = label_names[pred_idx]

        status = "✅ Doğru" if true_idx == pred_idx else "❌ Yanlış"

        print(f"{i:<4} {true_name:<20} {pred_name:<20} {status:<10}")

    print("-" * 60)
    print("Not: Bu sadece ilk batch'ten birkaç örnek, tüm test seti değil.\n")


if __name__ == "__main__":
    demo_on_test_samples(num_samples=10)

