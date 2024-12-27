import os
import shutil
import random

# Path ke dataset asli
SOURCE_DIR = "../../dataset/Insect Classes"
TARGET_DIR = "../../dataset/classification"
SPLIT_RATIO = 0.8  # 80% untuk train, 20% untuk val

def split_dataset():
    for cls in os.listdir(SOURCE_DIR):
        class_dir = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(class_dir):
            continue

        # Buat folder `train/` dan `val/` untuk setiap kelas
        train_dir = os.path.join(TARGET_DIR, "train", cls)
        val_dir = os.path.join(TARGET_DIR, "val", cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Ambil semua file dalam kelas dan acak
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(files)

        # Hitung batas untuk split
        split_point = int(len(files) * SPLIT_RATIO)

        # Pindahkan file ke folder `train/` dan `val/`
        for i, file in enumerate(files):
            src_file = os.path.join(class_dir, file)
            if i < split_point:
                shutil.copy(src_file, os.path.join(train_dir, file))
            else:
                shutil.copy(src_file, os.path.join(val_dir, file))

    print(f"Dataset berhasil dipisahkan ke dalam folder: {TARGET_DIR}")

# Panggil fungsi
split_dataset()
