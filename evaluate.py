import os
import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from data.dataset import StrokeDataset
from models.biomedclip_encoder import ImageEncoderWithMLP

val_csv = "data/data_paths/test_data.csv"
model_path = "checkpoints\stroke_model_iter006.pt"
log_path = "evaluation_logs.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df["filepath"].tolist(), df["label"].tolist()

val_paths, val_labels = load_data(val_csv)

model = ImageEncoderWithMLP(
    config_path="configs/open_clip_config.json",
    checkpoint_path="checkpoints/open_clip_pytorch_model.bin"
).to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

for param in model.encoder.parameters():
    param.requires_grad = False


val_dataset = StrokeDataset(val_paths, val_labels, transform=model.get_preprocess())
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


all_preds, all_labels, all_confs = [], [], []

with torch.no_grad():
    for images, labels_batch in tqdm(val_loader):
        images = images.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)

        outputs = model(images)
        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        confs = np.abs(probs - 0.5) * 2

        all_preds.extend(preds.flatten())
        all_labels.extend(labels_batch.cpu().numpy().flatten())
        all_confs.extend(confs.flatten())


accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
roc_auc = roc_auc_score(all_labels, all_preds)
confidence = np.mean(all_confs)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print(f"Avg Conf : {confidence:.4f}")


write_header = not os.path.exists(log_path)
with open(log_path, mode="a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "avg_confidence"
        ])
    writer.writerow([
        os.path.basename(model_path), accuracy, precision, recall, f1, roc_auc, confidence
    ])


cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-Stroke", "Stroke"], yticklabels=["Non-Stroke", "Stroke"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
