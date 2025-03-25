import os
import glob
import random
import pandas as pd

# Klasör yolları
base_dir = "Brain_Data_Organised"
class_dirs = {
    "Stroke": 1,
    "Normal": 0
}

# Tüm görüntü yolları ve etiketleri topla
image_paths = []
labels = []

for subdir, label in class_dirs.items():
    full_path = os.path.join(base_dir, subdir)
    png_files = glob.glob(os.path.join(full_path, "*.jpg"))
    image_paths.extend(png_files)
    labels.extend([label] * len(png_files))

# Shuffle + split
combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths, labels = zip(*combined)

# CSV kaydetme fonksiyonu
def save_csv(paths, labels, filename):
    df = pd.DataFrame({"filepath": paths, "label": labels})
    df.to_csv(filename, index=False)

os.makedirs("data/data_paths", exist_ok=True)
save_csv(image_paths, labels, "data/data_paths/kaggle_test_data.csv")

print(f"Toplam: {len(image_paths)}")