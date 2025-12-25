import random
import torch
from torchvision import transforms
from PIL import Image
import gradio as gr
from datasets import load_dataset

from src.models_baseline import SimpleCNN
from src.config import IMAGE_SIZE


CLASS_NAMES = [
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



def load_model():

    model = SimpleCNN()
    state_dict = torch.load("baseline_improved_best.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = load_model()



try:
    print("✅ EuroSAT_RGB HuggingFace dataseti yükleniyor...")
    hf_ds = load_dataset("blanchon/EuroSAT_RGB", split="train")
    print("Toplam örnek sayısı:", len(hf_ds))
except Exception as e:
    print("❌ HuggingFace dataseti yüklenirken hata:", e)
    hf_ds = None


def load_random_hf_image():

    if hf_ds is None:
        raise gr.Error(
            "HuggingFace EuroSAT_RGB datasetine erişilemiyor. "
            "İnternet bağlantını veya HuggingFace erişimini kontrol et."
        )

    idx = random.randint(0, len(hf_ds) - 1)
    sample = hf_ds[idx]
    img = sample["image"]  
    return img



transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
])


def predict(image: Image.Image):
    if image is None:
        raise gr.Error("Önce bir görsel yükle veya HuggingFace butonuna bas.")

    img = transform(image).unsqueeze(0)  # [1, C, H, W]

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = CLASS_NAMES[pred_idx]

    prob_dict = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return pred_class, prob_dict



with gr.Blocks() as demo:
    gr.Markdown("# EuroSAT LULC Sınıflandırma (Improved Baseline CNN)")
    gr.Markdown(
        "Soldan **manuel görsel yükleyebilir** veya sağdaki butonla "
        "**HuggingFace EuroSAT_RGB datasından rastgele bir örnek** getirebilirsin."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Uydu Görselini Yükle", interactive=True)

        with gr.Column():
            hf_btn = gr.Button("HuggingFace'den Rastgele Görsel Getir")
            predict_btn = gr.Button("Tahmin Et")

    pred_label = gr.Label(label="Tahmin Edilen Sınıf")
    probs_label = gr.Label(label="Sınıf Olasılıkları")


    hf_btn.click(
        fn=load_random_hf_image,
        inputs=None,
        outputs=image_input,
    )


    predict_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[pred_label, probs_label],
    )


if __name__ == "__main__":
    demo.launch()
