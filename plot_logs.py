import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 🔧 CSV dosyasının yolu (istediğini buradan seç)
#log_path = "logs/training_logs.csv"

log_base = "logs/2025_03_25__14_11_01"

log_name = "training_iter_details.csv"

log_path = os.path.join(log_base, log_name)

# log_path = "logs/evaluation_logs.csv"

assert os.path.exists(log_path), f"{log_path} bulunamadı!"

# CSV'yi oku
df = pd.read_csv(log_path)

# X ekseni olarak ilk sütunu al (epoch, iter, model vs)
x_col = df.columns[0]
y_cols = df.columns[1:]



"""
['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background',
'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'petroff10', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind',
'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook',
'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid',
'tableau-colorblind10']
"""

plot_timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

os.makedirs(os.path.join(log_base, f"plots/{plot_timestamp}"), exist_ok=True)

plot_dir_path = os.path.join(log_base, f"plots/{plot_timestamp}")

# plt.style.use("ggplot")
# Her metrik için ayrı grafik çiz
for col in y_cols:
    plt.figure()
    plt.plot(df[x_col], df[col])
    plt.title(f"{col} vs {x_col}")
    plt.xlabel(x_col)
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{plot_dir_path}/{col}_{plot_timestamp}.png")
    plt.close()