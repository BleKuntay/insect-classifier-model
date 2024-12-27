import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Path dataset dan model
VAL_DIR = "../../dataset/classification/val"
MODEL_PATH = "../../models/densenet121_best.pth"
DEVICE = torch.device("cpu")  # Paksa menggunakan CPU

# Transformasi Data
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset dan DataLoader
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Load Model
model = models.densenet121(pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, len(val_dataset.classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Tambahkan `map_location` untuk memastikan model di-load ke CPU
model = model.to(DEVICE)
model.eval()

# Validasi Model
def validate_model():
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)  # Pastikan tensor dipindahkan ke CPU

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Hitung akurasi
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    validate_model()
