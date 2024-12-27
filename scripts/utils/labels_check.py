# Contoh: Pytorch DataLoader
from torchvision import datasets
from torchvision.transforms import transforms

data_dir = '../../dataset/classification/train'
train_dataset = datasets.ImageFolder(data_dir, transform=transforms)
print(train_dataset.class_to_idx)
