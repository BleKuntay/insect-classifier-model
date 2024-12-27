import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Path ke dataset
TRAIN_DIR = "../../dataset/classification/train"
VAL_DIR = "../../dataset/classification/val"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = len(os.listdir(TRAIN_DIR))  # Jumlah kelas dari folder train

# Data Augmentation dan Normalisasi
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Augmentasi data
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset dan DataLoader
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Model ResNet Pretrained
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # Ubah output sesuai jumlah kelas
model = model.to(DEVICE)

# Loss Function dan Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Untuk menyimpan metrik
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# Fungsi Training
def train_model():
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
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

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)
        print(f"Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")

        # Validasi setelah setiap epoch
        validate_model()

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

            # Hitung akurasi validasi
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    val_loss_history.append(avg_val_loss)
    val_acc_history.append(val_accuracy)
    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Plot Metrics
def plot_metrics():
    epochs_range = range(NUM_EPOCHS)

    plt.figure(figsize=(12, 6))

    # Plot Training and Validation Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc_history, label='Training Accuracy')
    plt.plot(epochs_range, val_acc_history, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # Plot Training and Validation Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss_history, label='Training Loss')
    plt.plot(epochs_range, val_loss_history, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.show()

# Panggil fungsi pelatihan
if __name__ == "__main__":
    train_model()
    plot_metrics()
    torch.save(model.state_dict(), "../../models/resnet50_classification.pth")
    print("Model ResNet50 berhasil disimpan di 'models/resnet50_classification.pth'")
