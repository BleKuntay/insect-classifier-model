import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

# Path dataset dan model
VAL_DIR = "../../dataset/classification/val"
MODEL_PATH = "../../models/resnet50_classification.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformasi Data
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset dan DataLoader
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load Model
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(val_dataset.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# Hitung Precision per Batch
def calculate_precision_per_batch():
    precisions = []
    batch_indices = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Hitung precision untuk batch ini
            precision = precision_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
            precisions.append(precision)
            batch_indices.append(batch_idx + 1)

    return batch_indices, precisions

# Plot Hasil Precision
def plot_precision(batch_indices, precisions):
    plt.figure(figsize=(10, 6))
    plt.plot(batch_indices, precisions, marker='o', linestyle='-', color='g', label="Precision")
    plt.ylim(0, 1)  # Precision dalam skala 0-1
    plt.xlabel("Batch Index")
    plt.ylabel("Precision")
    plt.title("Precision per Batch")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Tambahkan nilai pada setiap titik
    for i, v in zip(batch_indices, precisions):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)

    plt.show()

if __name__ == "__main__":
    batch_indices, precisions = calculate_precision_per_batch()

    print("Precision per batch:")
    for batch, precision in zip(batch_indices, precisions):
        print(f"Batch {batch}: {precision:.4f}")

    # Tampilkan grafik precision
    plot_precision(batch_indices, precisions)
