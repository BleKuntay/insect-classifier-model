import torch
from torchvision import transforms, models
from PIL import Image

# Path model dan gambar
MODEL_PATH = "models/mobilenet_v2_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Class1", "Class2", "Class3", "Class4"]  # Ubah sesuai kelas Anda

# Transformasi untuk gambar baru
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH))
model = model.to(DEVICE)
model.eval()

# Prediksi Gambar Baru
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = CLASS_NAMES[predicted.item()]
        return class_name

if __name__ == "__main__":
    image_path = input("Masukkan path gambar: ")
    predicted_class = predict_image(image_path)
    print(f"Predicted Class: {predicted_class}")
