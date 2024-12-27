import os
import shutil
import random

# Path ke dataset asli
SOURCE_DIR = "../../dataset/Insect Classes"
TARGET_DIR = "../../dataset/classification/test"
RATIO = 0.1  # 10% untuk sampel

def take_random_sample():
    for cls in os.listdir(SOURCE_DIR):
        class_dir = os.path.join(SOURCE_DIR, cls)
        if not os.path.isdir(class_dir):
            continue

        # Buat folder target untuk kelas
        sample_dir = os.path.join(TARGET_DIR, cls)
        os.makedirs(sample_dir, exist_ok=True)

        # Ambil semua file dalam kelas dan acak
        files = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        random.shuffle(files)

        # Hitung jumlah file yang akan diambil
        sample_count = max(1, int(len(files) * RATIO))

        # Pilih file secara acak dan pindahkan ke folder target
        for file in files[:sample_count]:
            src_file = os.path.join(class_dir, file)
            shutil.copy(src_file, os.path.join(sample_dir, file))

    print(f"Sampel 10% berhasil disalin ke folder: {TARGET_DIR}")

# Panggil fungsi
take_random_sample()
