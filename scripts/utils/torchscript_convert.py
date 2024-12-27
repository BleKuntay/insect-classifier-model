import torch
from torchvision.models import efficientnet_b4

# Inisialisasi model EfficientNet-B4
model = efficientnet_b4(pretrained=False)  # Jangan gunakan pretrained karena Anda akan memuat bobot sendiri

# Sesuaikan jumlah kelas di classifier
num_classes = 10  # Ubah sesuai kebutuhan Anda
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)

# Muat bobot model
state_dict = torch.load('../../models/fgg_efficientnet_b4_best.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set model ke mode evaluasi
model.eval()

# Konversi model ke TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

print("Model berhasil dimuat dan disimpan sebagai 'model_scripted.pt'")
