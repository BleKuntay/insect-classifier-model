import torch
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Daftar nama kelas
class_names = ['Bees', 'Beetles', 'Butterfly', 'Cicada', 'Dragonfly',
               'Grasshopper', 'Moth', 'Scorpion', 'Snail', 'Spider']

# Transformasi gambar (sesuai kebutuhan model)
resize_dim = (224, 224)
transform = transforms.Compose([
    transforms.Resize(resize_dim),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model klasifikasi
model = torch.jit.load("../../model_scripted.pt")
model.eval()

def predict_class(image_path):
    """Prediksi kelas dan confidence score untuk gambar."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class_idx = probabilities.argmax().item()
        confidence = probabilities[predicted_class_idx].item()

    return class_names[predicted_class_idx], confidence

def draw_classification_box(image_path, output_path="output.jpg"):
    """Gambar satu bounding box global dengan kelas dan confidence."""
    # Prediksi kelas
    class_name, confidence = predict_class(image_path)

    # Load gambar
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image.")
        return

    # Ukuran bounding box global
    height, width, _ = image.shape
    x_min, y_min, x_max, y_max = 50, 50, width - 50, height - 50

    # Gambar bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 0, 255)  # Red box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    text = f"{class_name}: {confidence:.2f}"
    cv2.putText(image, text, (x_min, y_min - 10), font, 0.7, color, 2)

    # Simpan dan tampilkan gambar
    cv2.imwrite(output_path, image)
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Dynamic Classification Results")
    plt.show()

# Contoh penggunaan
image_path = "../../dataset/external_images/Bee/bee4.webp"
draw_classification_box(image_path)
