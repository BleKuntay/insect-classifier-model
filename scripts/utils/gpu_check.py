import torch

def check_device():
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        print(f"GPU is available: {device_name}")
    else:
        print("GPU is not available. Using CPU.")

if __name__ == "__main__":
    check_device()
