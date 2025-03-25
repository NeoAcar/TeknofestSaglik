import os
import glob
import random
import pandas as pd
from sklearn.model_selection import train_test_split


label_df = pd.read_csv("labels.csv")

print(label_df.head())

# Klasör yolları
base_dir = "inme"
sub_dir = "external_clean"

# Tüm görüntü yolları ve etiketleri topla
image_paths = []
labels = []

full_path = os.path.join(base_dir, sub_dir)
png_files = glob.glob(os.path.join(full_path, "*.png"))
image_paths.extend(png_files)

print(len(png_files))

label_df["image_id"] = label_df["image_id"].astype(str)

for path in png_files:
    # get basename and get rid of extension
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    label = label_df[label_df["image_id"] == filename]["Stroke"].values[0]
    labels.append(label)
    
# # Shuffle + split
combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths, labels = zip(*combined)

# # Train / val / test split
# train_paths, temp_paths, train_labels, temp_labels = train_test_split(
#     image_paths, labels, test_size=0.3, random_state=42, stratify=labels
# )
# val_paths, test_paths, val_labels, test_labels = train_test_split(
#     temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
# )


# CSV kaydetme fonksiyonu
def save_csv(paths, labels, filename):
    df = pd.DataFrame({"filepath": paths, "label": labels})
    df.to_csv(filename, index=False)

os.makedirs("data/data_paths", exist_ok=True)
save_csv(image_paths, labels, "data/data_paths/external_test_data.csv")

print(f"Toplam: {len(image_paths)}")