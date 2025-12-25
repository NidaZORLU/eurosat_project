import re
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")

   
    train_re = re.compile(r"Train\s+Loss:\s*([0-9.]+)\s*\|\s*Train\s+Acc:\s*([0-9.]+)")
    val_re   = re.compile(r"Val\s+Loss:\s*([0-9.]+)\s*\|\s*Val\s+Acc:\s*([0-9.]+)")
    epoch_re = re.compile(r"Epoch\s+(\d+)/(\d+)")

    epochs = []
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    epoch_lines = epoch_re.findall(text)
    

    t = train_re.findall(text)
    v = val_re.findall(text)

    n = min(len(t), len(v))
    if n == 0:
        print(f"⚠️ Log formatı farklı, atlanıyor: {path.name}")
        return None


    for i in range(n):
        epochs.append(i + 1)
        train_loss.append(float(t[i][0]))
        train_acc.append(float(t[i][1]))
        val_loss.append(float(v[i][0]))
        val_acc.append(float(v[i][1]))

    

    test_re = re.compile(r"TEST.*Loss:\s*([0-9.]+)\s*\|\s*Acc(?:uracy)?:\s*([0-9.]+)")
    test = test_re.findall(text)
    test_loss = float(test[-1][0]) if test else None
    test_acc  = float(test[-1][1]) if test else None

    early_stop = "Early stopping tetiklendi" in text

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "early_stop": early_stop,
    }

def plot_curves(data, title_prefix, out_prefix):
    
    plt.figure()
    plt.plot(data["epochs"], data["train_loss"], label="Train Loss")
    plt.plot(data["epochs"], data["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} - Loss")
    plt.legend()
    plt.savefig(FIGURES / f"{out_prefix}_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ACC grafiği
    plt.figure()
    plt.plot(data["epochs"], data["train_acc"], label="Train Acc")
    plt.plot(data["epochs"], data["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{title_prefix} - Accuracy")
    plt.legend()
    plt.savefig(FIGURES / f"{out_prefix}_acc.png", dpi=200, bbox_inches="tight")
    plt.close()

def main():
    
    candidates = list(RESULTS.glob("*log*.txt"))
    if not candidates:
        raise FileNotFoundError("results/ içinde log*.txt bulunamadı.")

    print("Bulunan loglar:")
    for p in candidates:
        print(" -", p.name)

    for p in candidates:
        data = parse_log(p)
        if data is None:
            continue

        prefix = p.stem
        plot_curves(data, title_prefix=p.stem, out_prefix=prefix)

        print(f"\n[{p.name}]")
        print("  Epochs:", len(data["epochs"]))
        print("  Early stopping:", data["early_stop"])
        if data["test_acc"] is not None:
            print("  Test Acc:", data["test_acc"])
            print("  Test Loss:", data["test_loss"])

    print("\n✅ Grafikler figures/ klasörüne kaydedildi.")

if __name__ == "__main__":
    main()

