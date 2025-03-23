import os
import glob
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# Klasör yolları
base_dir = "inme"
class_dirs = {
    "var_iskemi/PNG": 1,
    "var_kanama/PNG": 1,
    "yok": 0
}

# Tüm görüntü yolları ve etiketleri topla
image_paths = []
labels = []

for subdir, label in class_dirs.items():
    full_path = os.path.join(base_dir, subdir)
    png_files = glob.glob(os.path.join(full_path, "*.png"))
    image_paths.extend(png_files)
    labels.extend([label] * len(png_files))

# Shuffle + split
combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths, labels = zip(*combined)

# Train / val / test split
train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, labels, test_size=0.3, random_state=42, stratify=labels
)
val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

# CSV kaydetme fonksiyonu
def save_csv(paths, labels, filename):
    df = pd.DataFrame({"filepath": paths, "label": labels})
    df.to_csv(filename, index=False)

os.makedirs("data/data_paths", exist_ok=True)
save_csv(train_paths, train_labels, "data/data_paths/train_data.csv")
save_csv(val_paths, val_labels, "data/data_paths/val_data.csv")
save_csv(test_paths, test_labels, "data/data_paths/test_data.csv")

print(f"Toplam: {len(image_paths)} | Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")