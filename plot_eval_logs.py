import pandas as pd
import matplotlib.pyplot as plt

eval_log_path = "evaluation_logs.csv"
df = pd.read_csv(eval_log_path)

plt.figure()
plt.plot(df["model"], df["accuracy"], marker="o")
plt.title("Accuracy per Model Checkpoint")
plt.xlabel("Model Checkpoint")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha="right")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df["model"], df["roc_auc"], marker="o", color='green')
plt.title("ROC AUC per Model Checkpoint")
plt.xlabel("Model Checkpoint")
plt.ylabel("ROC AUC")
plt.xticks(rotation=45, ha="right")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(df["model"], df["avg_confidence"], marker="o", color='orange')
plt.title("Average Confidence per Model Checkpoint")
plt.xlabel("Model Checkpoint")
plt.ylabel("Avg Confidence")
plt.xticks(rotation=45, ha="right")
plt.grid(True)
plt.tight_layout()
plt.show()
