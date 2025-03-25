import os
import csv
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from data.dataset import StrokeDataset
from data.transforms import get_stroke_dataset_transforms
from models.biomedclip_encoder import ImageEncoderWithMLP
import pandas as pd
from tqdm import tqdm
from datetime import datetime

train_csv = "data/data_paths/train_data_clean.csv"
val_csv = "data/data_paths/val_data_clean.csv"

timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

checkpoint_dir = f"checkpoints/{timestamp}"
log_dir = f"logs/{timestamp}"

os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

frozen_encoder = False
batch_size = 128
num_epochs = 20
lr = 1e-4
weight_decay = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df["filepath"].tolist(), df["label"].tolist()

train_paths, train_labels = load_data(train_csv)
val_paths, val_labels = load_data(val_csv)

model = ImageEncoderWithMLP(
    config_path="configs/open_clip_config.json",
    checkpoint_path="checkpoints/open_clip_pytorch_model.bin",
    frozen_encoder=frozen_encoder
).to(device)


if frozen_encoder:
    for param in model.encoder.parameters():
        param.requires_grad = False

custom_transforms = get_stroke_dataset_transforms()

train_dataset = StrokeDataset(train_paths, train_labels, transform=model.get_preprocess(), custom_transform=custom_transforms["train"])
val_dataset = StrokeDataset(val_paths, val_labels, transform=model.get_preprocess(), custom_transform=custom_transforms["val"])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)


log_path = f"{log_dir}/training_log.csv"
iter_details_path = f"{log_dir}/training_iter_details.csv"
write_header = not os.path.exists(log_path)

i = 0
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0
    all_preds, all_labels, all_confs = [], [], []
    total_norm = 0.0
    iter_details = []

    for images, labels_batch in tqdm(train_loader):
        i += 1
        images = images.to(device)
        labels_batch = labels_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()

        for p in model.classifier.parameters():
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
        batch_precision = precision_score(batch_labels, batch_preds, zero_division=0)
        batch_recall = recall_score(batch_labels, batch_preds, zero_division=0)
        batch_f1 = f1_score(batch_labels, batch_preds, zero_division=0)

        iter_details.append([
            i, batch_loss, batch_acc, grad_norm, np.mean(batch_conf), batch_roc_auc,
            batch_precision, batch_recall, batch_f1
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

    train_precision = precision_score(all_labels, all_preds, zero_division=0)
    train_recall = recall_score(all_labels, all_preds, zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, zero_division=0)

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

    val_precision = precision_score(val_labels_list, val_preds, zero_division=0)
    val_recall = recall_score(val_labels_list, val_preds, zero_division=0)
    val_f1 = f1_score(val_labels_list, val_preds, zero_division=0)

    scheduler.step(val_loss)

    print(
        f"Epoch {epoch:03d} | Train Loss: {epoch_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
        f"GradNorm: {grad_norm:.4f} | Train Conf: {train_conf:.4f} | Train AUC: {train_roc_auc:.4f} | "
        f"Train Precision: {train_precision:.4f} | Train Recall: {train_recall:.4f} | Train F1: {train_f1} | "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val Conf: {val_conf:.4f} | Val AUC: {val_roc_auc:.4f} | "
        f"Val Precision: {val_precision:.4f} | Val Recall: {val_recall:.4f} | Val F1: {val_f1:.4f} | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
    )

    with open(log_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch", "train_loss", "train_acc", "grad_norm", "train_conf", "train_auc",
                "train_precision", "train_recall", "train_f1",
                "val_loss", "val_acc", "val_conf", "val_auc",
                "val_precision", "val_recall", "val_f1",
                "lr"
            ])
            write_header = False
        writer.writerow([
            epoch, epoch_loss, train_accuracy, grad_norm, train_conf, train_roc_auc,
            train_precision, train_recall, train_f1,
            val_loss, val_accuracy, val_conf, val_roc_auc,
            val_precision, val_recall, val_f1,
            optimizer.param_groups[0]['lr']
        ])

    write_iter_header = not os.path.exists(iter_details_path)
    with open(iter_details_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if write_iter_header:
            writer.writerow([
                "iter", "train_loss", "train_acc", "grad_norm", "train_conf", "train_auc",
                "train_precision", "train_recall", "train_f1"
            ])
            write_iter_header = False
        writer.writerows(iter_details)

    torch.save(model.state_dict(), f"{checkpoint_dir}/stroke_model_iter{epoch:03d}.pt")
