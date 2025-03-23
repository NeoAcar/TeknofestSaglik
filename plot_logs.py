import pandas as pd
import matplotlib.pyplot as plt
import os

# 🔧 CSV dosyasının yolu (istediğini buradan seç)
#log_path = "logs/training_logs.csv"
log_path = "logs/2025_03_23__20_35_15/training_iter_details.csv"
# log_path = "logs/evaluation_logs.csv"

assert os.path.exists(log_path), f"{log_path} bulunamadı!"

# CSV'yi oku
df = pd.read_csv(log_path)

# X ekseni olarak ilk sütunu al (epoch, iter, model vs)
x_col = df.columns[0]
y_cols = df.columns[1:]

# Her metrik için ayrı grafik çiz
for col in y_cols:
    plt.figure()
    plt.plot(df[x_col], df[col], marker="o")
    plt.title(f"{col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
