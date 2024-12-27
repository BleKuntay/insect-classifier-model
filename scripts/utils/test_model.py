import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Daftar kelas
class_names = ['Bees', 'Beetles', 'Butterfly', 'Cicada', 'Dragonfly',
               'Grasshopper', 'Moth', 'Scorpion', 'Snail', 'Spider']

# Transformasi gambar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
model = torch.jit.load("../../model_scripted.pt")
model.eval()

def predict_single_image(image_path):
    """Prediksi confidence untuk satu gambar."""
    try:
        # Load dan transform gambar
        image = Image.open(image_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Prediksi
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence = probabilities.max().item()  # Ambil confidence tertinggi

        return round(confidence * 100, 2)  # Kembalikan confidence dalam persen
    except Exception as e:
        print(f"Error: {e}")
        return None


def plot_confidence_per_class(test_folder):
    """Membuat grafik confidence naik-turun untuk setiap kelas."""
    for cls in class_names:
        class_path = os.path.join(test_folder, cls)
        if not os.path.isdir(class_path):
            print(f"Warning: Folder {cls} tidak ditemukan!")
            continue

        print(f"Processing class: {cls}")
        confidences = []

        # Loop semua gambar dalam folder kelas
        for image_file in sorted(os.listdir(class_path)):
            if image_file.endswith(('jpg', 'jpeg', 'png', 'webp')):
                image_path = os.path.join(class_path, image_file)
                confidence = predict_single_image(image_path)
                if confidence is not None:
                    confidences.append(confidence)

        # Plot grafik confidence untuk kelas ini
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(confidences)), confidences, marker='o', linestyle='-', color='b')
        plt.title(f"Confidence Scores - {cls}")
        plt.xlabel("Image Index")
        plt.ylabel("Confidence (%)")
        plt.ylim(0, 100)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Folder test dataset
test_folder = "../../dataset/classification/test"

# Jalankan fungsi untuk membuat grafik
plot_confidence_per_class(test_folder)
