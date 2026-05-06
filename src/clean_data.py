import pandas as pd
import os

# Giriş ve çıkış yolları
input_path = "data/processed/anomaly_dataset.csv"
output_path = "data/processed/anomaly_dataset_clean.csv"

print("Dosya var mı:", os.path.exists(input_path))

# Veriyi oku
df = pd.read_csv(input_path)

print("\nOrijinal kolonlar:")
print(df.columns)

# ----------------------------
# LEAKAGE KOLONLARI SİL
# ----------------------------
drop_cols = [
    "anomaly_type",
    "Appliances_Original"
]

df_clean = df.drop(columns=drop_cols, errors="ignore")

# ----------------------------
# DATE KORUNSUN AMA MODELDE KULLANILMASIN
# ----------------------------

# date'i datetime yap (grafik için lazım)
df_clean["date"] = pd.to_datetime(df_clean["date"])

# ----------------------------
# MODEL İÇİN SAYISAL KOLONLAR
# ----------------------------

# Model için ayrı X oluşturacağız (date hariç)
numeric_cols = df_clean.select_dtypes(include=["number"]).columns.tolist()

# Hedefi de dahil ediyoruz
if "is_anomaly" not in numeric_cols:
    numeric_cols.append("is_anomaly")

# Model datası (date hariç)
df_model = df_clean[numeric_cols]

# ----------------------------
# KAYDET
# ----------------------------

df_clean.to_csv(output_path, index=False)

print("\nTemiz veri kaydedildi:", output_path)

print("\nYeni kolonlar:")
print(df_clean.columns)

print("\nVeri dağılımı:")
print(df_clean["is_anomaly"].value_counts())