import torch
import torch.nn as nn
import torch.optim as optim

from .config import DEVICE, LEARNING_RATE, WEIGHT_DECAY, NUM_EPOCHS, set_seed
from .dataset_eurosat import get_dataloaders
from .models_baseline import SimpleCNN
from .utils import train_one_epoch, eval_one_epoch


def main():
    
    set_seed()

    print(f"\n🚀 Eğitim başlıyor | Kullanılan cihaz: {DEVICE}\n")

    train_loader, val_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_acc = 0.0


    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n====== Epoch {epoch}/{NUM_EPOCHS} ======\n")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )

        val_loss, val_acc = eval_one_epoch(
            model, val_loader, criterion, DEVICE, desc="Validation"
        )

        print(f"🟩 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"🟦 Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "baseline_best.pth")
            print("\n💾 Yeni en iyi model kaydedildi: baseline_best.pth\n")


    print("\n🔍 En iyi modeli test seti ile değerlendiriyoruz...\n")
    model.load_state_dict(torch.load("baseline_best.pth", map_location=DEVICE))

    test_loss, test_acc = eval_one_epoch(
        model, test_loader, criterion, DEVICE, desc="Test"
    )

    print(f"\n🎯 TEST Sonuçları → Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}\n")


if __name__ == "__main__":
    main()

