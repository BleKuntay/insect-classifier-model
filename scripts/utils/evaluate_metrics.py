import torch

# Muat model TorchScript
scripted_model = torch.jit.load("../../model_scripted.pt")

# Set ke mode evaluasi (opsional jika sudah diekspor dalam mode eval)
scripted_model.eval()
