import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Path dataset dan model
VAL_DIR = "../../dataset/classification/val"
MODEL_PATH = "../../models/fgg_efficientnet_b4_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformasi Data
val_transform = transforms.Compose([
    transforms.Resize((380, 380)),  # Sesuaikan dengan EfficientNet-B4
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset dan DataLoader
try:
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
except FileNotFoundError as e:
    print(f"Error: Dataset not found at {VAL_DIR}. Please check the path.")
    exit()

# Load Model
try:
    model = models.efficientnet_b4(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(val_dataset.classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Support for both GPU and CPU
    model = model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error: Failed to load the model. Details: {e}")
    exit()

# Validasi Model
def validate_model():
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

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
    try:
        validate_model()
    except Exception as e:
        print(f"Error during validation. Details: {e}")
