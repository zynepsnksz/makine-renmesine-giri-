import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/processed/feature_engineered_data.csv")

df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)

df_subset = df.loc["2016-01-12":"2016-01-18"].copy()

df_subset["is_anomaly"] = 0
df_subset["Appliances_Original"] = df_subset["Appliances"].copy()

# --- Senaryo 1: Tost makinesi (kısa spike)
anomaly_start_1 = pd.to_datetime("2016-01-14 02:00:00")
periods_1 = 4
extra_watt_1 = 800

for i in range(periods_1):
    current_time = anomaly_start_1 + pd.Timedelta(minutes=10 * i)
    if current_time in df_subset.index:
        df_subset.loc[current_time, "Appliances"] += extra_watt_1
        df_subset.loc[current_time, "is_anomaly"] = 1
        df_subset.loc[current_time, "anomaly_type"] = "Tost makinesi açık unutuldu"

# --- Senaryo 2: Kombi arızası (uzun yüksek)
anomaly_start_2 = pd.to_datetime("2016-01-16 14:30:00")
periods_2 = 6
extra_watt_2 = 600

for i in range(periods_2):
    current_time = anomaly_start_2 + pd.Timedelta(minutes=10 * i)
    if current_time in df_subset.index:
        df_subset.loc[current_time, "Appliances"] += extra_watt_2
        df_subset.loc[current_time, "is_anomaly"] = 1
        df_subset.loc[current_time, "anomaly_type"] = "Kombi/ısıtıcı arızası"

df_subset["anomaly_type"] = df_subset["anomaly_type"].fillna("Normal")

# --- Görselleştirme
plt.figure(figsize=(15, 6))

plt.plot(
    df_subset.index,
    df_subset["Appliances_Original"],
    label="Orijinal tüketim",
    alpha=0.6
)

plt.plot(
    df_subset.index,
    df_subset["Appliances"],
    label="Anomalili tüketim",
    alpha=0.7,
    linestyle="--"
)

anomalies = df_subset[df_subset["is_anomaly"] == 1]

plt.scatter(
    anomalies.index,
    anomalies["Appliances"],
    s=50,
    label="Anomali",
    zorder=5
)

plt.title("Sentetik Anomali Enjeksiyonu")
plt.xlabel("Tarih")
plt.ylabel("Enerji Tüketimi")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()

plt.savefig("outputs/anomaly_visualization.png", dpi=300)

# --- CSV'yi DOĞRU isimle kaydet
df_subset.to_csv("data/processed/anomaly_dataset.csv")

print("Anomali enjeksiyonu tamamlandı.")
print("Grafik kaydedildi: outputs/anomaly_visualization.png")
print("Anomalili veri kaydedildi: data/processed/anomaly_dataset.csv")
print(df_subset["is_anomaly"].value_counts())