import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Path dataset dan model
VAL_DIR = "../../dataset/classification/val"  # Pastikan path ini benar
MODEL_PATH = "../../models/mobilenet_v2_best.pth"  # Pastikan file model tersedia
DEVICE = torch.device("cpu")  # Paksa penggunaan CPU

# Transformasi Data
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # MobileNetV2 membutuhkan input 224x224
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalisasi standar untuk pretrained model
])

# Dataset dan DataLoader
try:
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
except FileNotFoundError:
    print(f"Error: Dataset not found at {VAL_DIR}. Please check the path.")
    exit()

# Load Model
try:
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(val_dataset.classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Paksa semua operasi di CPU
    model = model.to(DEVICE)  # Pastikan model dialokasikan ke CPU
    model.eval()
except Exception as e:
    print(f"Error loading model. Details: {e}")
    exit()

# Validasi Model
def validate_model():
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Nonaktifkan grad untuk validasi
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)  # Pastikan semua data di CPU

            outputs = model(images)  # Inference
            loss = criterion(outputs, labels)  # Hitung loss
            val_loss += loss.item()

            # Hitung akurasi
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    try:
        validate_model()
    except Exception as e:
        print(f"Error during validation. Details: {e}")
