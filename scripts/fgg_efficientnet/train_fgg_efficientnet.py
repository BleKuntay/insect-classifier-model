import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time  # Untuk menghitung durasi waktu

# Path ke dataset
TRAIN_DIR = "../../dataset/classification/train"
VAL_DIR = "../../dataset/classification/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 8  # Batch kecil karena FGG membutuhkan resolusi tinggi
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001  # Fine-tuning membutuhkan learning rate kecil
NUM_CLASSES = len(os.listdir(TRAIN_DIR))  # Jumlah kelas

# Data Augmentation dan Normalisasi
train_transform = transforms.Compose([
    transforms.Resize((380, 380)),  # Resolusi tinggi (untuk EfficientNet-B4 ke atas)
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),  # Augmentasi lebih kompleks
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalisasi pretrained model
])

val_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset dan DataLoader
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Model EfficientNet Pretrained
model = models.efficientnet_b4(pretrained=True)  # Menggunakan EfficientNet-B4
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(DEVICE)

# Loss Function dan Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)  # AdamW stabil untuk fine-tuning

# History untuk plotting
history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

# Fungsi Training
def train_model():
    best_val_accuracy = 0.0

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()  # Waktu mulai epoch
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass dan Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Hitung akurasi training
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Cetak progress batch dalam persen
            progress = (batch_idx + 1) / len(train_loader) * 100
            print(f"\rProgress: {progress:.2f}%", end="")

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        print(f"\nTraining Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Validasi setelah setiap epoch
        val_loss, val_accuracy = validate_model()

        # Simpan History
        history["train_loss"].append(avg_train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        # Simpan model terbaik berdasarkan akurasi validasi
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "../../models/fgg_efficientnet_b4_best.pth")
            print(f"Model terbaik disimpan dengan akurasi validasi: {best_val_accuracy:.2f}%")

        # Hitung dan cetak waktu epoch
        epoch_duration = time.time() - start_time
        print(f"Epoch {epoch+1} selesai dalam waktu: {epoch_duration:.2f} detik\n")

    # Plot hasil training
    plot_training_history(history)

# Fungsi Validasi
def validate_model():
    model.eval()
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
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return avg_val_loss, accuracy

# Fungsi Plot History Training dan Validation
def plot_training_history(history):
    epochs_range = range(1, NUM_EPOCHS + 1)

    # Plot Akurasi
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_accuracy"], label="Training Accuracy")
    plt.plot(epochs_range, history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_loss"], label="Training Loss")
    plt.plot(epochs_range, history["val_loss"], label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    plt.show()

# Panggil fungsi pelatihan
if __name__ == "__main__":
    train_model()