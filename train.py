import os
import csv
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from data.dataset import StrokeDataset
from models.biomedclip_encoder import ImageEncoderWithMLP
import pandas as pd
from tqdm import tqdm

print("####################################################################################################")
train_csv = "data/data_paths/train_data.csv"
val_csv = "data/data_paths/val_data.csv"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)

batch_size = 64
num_epochs = 20
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df["filepath"].tolist(), df["label"].tolist()

train_paths, train_labels = load_data(train_csv)
val_paths, val_labels = load_data(val_csv)

print("train and validation data loaded")

model = ImageEncoderWithMLP(
    config_path="configs/open_clip_config.json",
    checkpoint_path="checkpoints/open_clip_pytorch_model.bin"
).to(device)

print("model initialized")

for param in model.encoder.parameters():
    param.requires_grad = False

print("encoder layers frozen")

train_dataset = StrokeDataset(train_paths, train_labels, transform=model.get_preprocess())
val_dataset = StrokeDataset(val_paths, val_labels, transform=model.get_preprocess())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

log_path = "logs/training_logs.csv"
iter_details_path = "logs/training_iter_details.csv"
write_header = not os.path.exists(log_path)

print("train cycle started")
i = 0
for epoch in range(1, num_epochs + 1):
    i += 1
    model.train()
    epoch_loss = 0
    all_preds, all_labels, all_confs = [], [], []
    total_norm = 0.0
    iter_details = []

    for images, labels_batch in tqdm(train_loader):
        images = images.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        optimizer.step()

        batch_loss = loss.item()
        probs = torch.sigmoid(outputs.detach()).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        confs = np.abs(probs - 0.5) * 2
        grad_norm = total_norm ** 0.5

        batch_preds = preds.flatten()
        batch_labels = labels_batch.cpu().numpy().flatten()
        batch_conf = confs.flatten()

        try:
            batch_roc_auc = roc_auc_score(batch_labels, batch_preds)
        except:
            batch_roc_auc = 0.0

        batch_acc = accuracy_score(batch_labels, batch_preds)

        iter_details.append([
            i, batch_loss, batch_acc, grad_norm, np.mean(batch_conf), batch_roc_auc
        ])

        epoch_loss += batch_loss * images.size(0)
        all_preds.extend(batch_preds)
        all_labels.extend(batch_labels)
        all_confs.extend(batch_conf)

    epoch_loss /= len(train_dataset)
    train_accuracy = accuracy_score(all_labels, all_preds)
    train_conf = np.mean(all_confs)
    grad_norm = total_norm ** 0.5
    try:
        train_roc_auc = roc_auc_score(all_labels, all_preds)
    except:
        train_roc_auc = 0.0

    model.eval()
    val_loss = 0
    val_preds, val_labels_list, val_confs = [], [], []

    with torch.no_grad():
        for images, labels_batch in val_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            confs = np.abs(probs - 0.5) * 2

            val_preds.extend(preds)
            val_labels_list.extend(labels_batch.cpu().numpy())
            val_confs.extend(confs)

    val_loss /= len(val_dataset)
    val_accuracy = accuracy_score(val_labels_list, val_preds)
    val_conf = np.mean(val_confs)
    try:
        val_roc_auc = roc_auc_score(val_labels_list, val_preds)
    except:
        val_roc_auc = 0.0

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
        f"GradNorm: {grad_norm:.4f} | Train Conf: {train_conf:.4f} | Train AUC: {train_roc_auc:.4f} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val Conf: {val_conf:.4f} | Val AUC: {val_roc_auc:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']} "
    )

    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch", "train_loss", "train_acc", "grad_norm", "train_conf", "train_auc",
                "val_loss", "val_acc", "val_conf", "val_auc", "LR"
            ])
            write_header = False
        writer.writerow([
            epoch, epoch_loss, train_accuracy, grad_norm, train_conf, train_roc_auc,
            val_loss, val_accuracy, val_conf, val_roc_auc, optimizer.param_groups[0]['lr']
        ])

    write_iter_header = not os.path.exists(iter_details_path)
    with open(iter_details_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_iter_header:
            writer.writerow([
                "iter","train_loss", "train_acc", "grad_norm", "train_conf", "train_auc"
            ])
            write_iter_header = False
        writer.writerows(iter_details)

    torch.save(model.state_dict(), f"{checkpoint_dir}/stroke_model_iter{epoch:03d}.pt")
