import torch

print("PyTorch CUDA Version:", torch.version.cuda)  # Versi CUDA yang didukung oleh PyTorch
print("Is CUDA available:", torch.cuda.is_available())  # Apakah CUDA tersedia
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))  # Nama GPU yang digunakan

