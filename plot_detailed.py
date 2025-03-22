import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/training_iter_details.csv")

fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 18), sharex=True)

axes[0].plot(df["train_loss"], label="Train Loss", color="tab:blue")
axes[0].set_title("Train Loss (per iteration)")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df["train_acc"], label="Train Accuracy", color="tab:green")
axes[1].set_title("Train Accuracy (per iteration)")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True)

axes[2].plot(df["grad_norm"], label="Gradient Norm", color="tab:red")
axes[2].set_title("Gradient Norm (per iteration)")
axes[2].set_ylabel("Grad Norm")
axes[2].legend()
axes[2].grid(True)

axes[3].plot(df["train_conf"], label="Confidence", color="tab:orange")
axes[3].set_title("Confidence (per iteration)")
axes[3].set_ylabel("Confidence")
axes[3].legend()
axes[3].grid(True)

axes[4].plot(df["train_auc"], label="Train ROC AUC", color="tab:purple")
axes[4].set_title("ROC AUC (per iteration)")
axes[4].set_ylabel("AUC")
axes[4].set_xlabel("Iteration")
axes[4].legend()
axes[4].grid(True)

plt.tight_layout()
plt.show()
