import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import os

from models.biomedclip_encoder import ImageEncoderWithMLP
from data.dataset import StrokeDataset


config_path = "configs/biomedclip_config.json"
checkpoint_path = "checkpoints/open_clip_pytorch_model.bin"
model_weights = "checkpoints/stroke_model_epoch20.pt"  # örnek
image_paths = [...]  # test görüntüleri
labels = [...]        # test etiketleri

batch_size = 8
log_csv_path = "evaluation_logs.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


model = ImageEncoderWithMLP(config_path, checkpoint_path).to(device)
model.load_state_dict(torch.load(model_weights, map_location=device))
model.eval()


dataset = StrokeDataset(image_paths, labels, transform=model.get_preprocess())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels_batch in dataloader:
        images = images.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)

        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)

        all_preds.extend(preds)
        all_probs.extend(probs)
        all_labels.extend(labels_batch.cpu().numpy())


acc = accuracy_score(all_labels, all_preds)
roc_auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

print(f"Accuracy: {acc:.4f}")
print(f"ROC AUC:  {roc_auc:.4f}")
print("Confusion Matrix:")
print(cm)


confidences = np.abs(np.array(all_probs) - 0.5) * 2
plt.hist(confidences, bins=20, color='skyblue', edgecolor='black')
plt.title("Model Confidence Distribution")
plt.xlabel("Eminlik")
plt.ylabel("Örnek Sayısı")
plt.grid(True)
plt.tight_layout()
plt.show()


sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()


header = ["model", "accuracy", "roc_auc", "avg_confidence"]
data_row = [os.path.basename(model_weights), f"{acc:.4f}", f"{roc_auc:.4f}", f"{np.mean(confidences):.4f}"]

write_header = not os.path.exists(log_csv_path)
with open(log_csv_path, mode="a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow(header)
    writer.writerow(data_row)
