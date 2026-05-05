import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

np.random.seed(42)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("data/processed/feature_engineered_data.csv")

df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

df["is_anomaly"] = 0
df["anomaly_type"] = "Normal"
df["Appliances_Original"] = df["Appliances"].copy()

appliances_col = df.columns.get_loc("Appliances")
is_anomaly_col = df.columns.get_loc("is_anomaly")
anomaly_type_col = df.columns.get_loc("anomaly_type")

# 1. Kısa süreli yüksek tüketim: ütü / tost makinesi açık kaldı
spike_indexes = np.random.choice(len(df), size=120, replace=False)

for i in spike_indexes:
    df.iloc[i, appliances_col] += 700
    df.iloc[i, is_anomaly_col] = 1
    df.iloc[i, anomaly_type_col] = "Kisa sureli cihaz acik kaldi"

# 2. Uzun süreli yüksek tüketim: kombi / ısıtıcı problemi
for _ in range(6):
    start = np.random.randint(0, len(df) - 36)
    end = start + 36  # 36 satır = 6 saat, çünkü veri 10 dakikalık

    df.iloc[start:end, appliances_col] += 350
    df.iloc[start:end, is_anomaly_col] = 1
    df.iloc[start:end, anomaly_type_col] = "Uzun sureli isitici problemi"

# Grafik için ilk 1000 satır yerine anomali içeren bir bölüm seçelim
first_anomaly_index = df.index[df["is_anomaly"] == 1][0]
plot_start = first_anomaly_index - pd.Timedelta(hours=12)
plot_end = first_anomaly_index + pd.Timedelta(hours=24)

df_plot = df.loc[plot_start:plot_end]

plt.figure(figsize=(15, 5))

plt.plot(
    df_plot.index,
    df_plot["Appliances_Original"],
    label="Orijinal tüketim",
    alpha=0.6
)

plt.plot(
    df_plot.index,
    df_plot["Appliances"],
    label="Anomalili tüketim",
    linestyle="--",
    alpha=0.8
)

anomalies = df_plot[df_plot["is_anomaly"] == 1]

plt.scatter(
    anomalies.index,
    anomalies["Appliances"],
    label="Anomali",
    s=40
)

plt.title("Sentetik Anomali Enjeksiyonu")
plt.xlabel("Tarih")
plt.ylabel("Enerji Tüketimi")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()

plt.savefig("outputs/anomaly_visualization.png", dpi=300)
plt.close()

df.to_csv("data/processed/anomaly_dataset.csv")

print("Anomali dataset oluşturuldu.")
print(df["is_anomaly"].value_counts())
print(df["anomaly_type"].value_counts())
print("Kaydedilen dosya: data/processed/anomaly_dataset.csv")
print("Kaydedilen grafik: outputs/anomaly_visualization.png")