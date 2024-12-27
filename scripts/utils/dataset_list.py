import os

# Path ke folder training
train_dir = "../../dataset/classification/train"

# Urutkan nama folder sesuai abjad untuk konsistensi
class_names = sorted(os.listdir(train_dir))

print("Class names and indices:")
for idx, class_name in enumerate(class_names):
    print(f"{idx}: {class_name}")
