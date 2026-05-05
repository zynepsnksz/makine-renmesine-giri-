import pandas as pd

# Ham veriyi oku
df = pd.read_csv("data/raw/energydata_complete.csv")

# Tarih sütununu datetime formatına çevir
df["date"] = pd.to_datetime(df["date"])

# ----------------------------
# 1. ZAMAN ÖZELLİĞİ
# ----------------------------

# Saat bilgisi: Gün içindeki tüketim davranışını yakalar
df["Hour"] = df["date"].dt.hour

# Hafta sonu bilgisi: Hafta içi / hafta sonu tüketim farkını yakalar
df["Is_Weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

# ----------------------------
# 2. SICAKLIK ÖZELLİKLERİ
# ----------------------------

temperature_columns = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]

# Evin genel iç sıcaklığı
df["Average_Indoor_Temperature"] = df[temperature_columns].mean(axis=1)

# İç ortam ile dış ortam sıcaklık farkı
df["Delta_T"] = df["Average_Indoor_Temperature"] - df["T_out"]

# ----------------------------
# 3. GEÇMİŞ TÜKETİM ÖZELLİKLERİ
# ----------------------------

# Bir önceki enerji tüketimi
df["Energy_Lag_1"] = df["Appliances"].shift(1)

# Son 6 ölçümün ortalama tüketimi
# Veri 10 dakikalık olduğu için 6 ölçüm yaklaşık 1 saate karşılık gelir
df["Energy_Rolling_Mean"] = df["Appliances"].rolling(window=6).mean()

# Son 6 ölçümdeki dalgalanma
df["Energy_Rolling_Std"] = df["Appliances"].rolling(window=6).std()

# Boş satırları temizle
df = df.dropna()

# İşlenmiş veriyi kaydet
df.to_csv("data/processed/feature_engineered_data.csv", index=False)

print("Feature engineering tamamlandı.")
print("Yeni veri boyutu:", df.shape)

print(df[[
    "date",
    "Appliances",
    "Hour",
    "Is_Weekend",
    "Average_Indoor_Temperature",
    "Delta_T",
    "Energy_Lag_1",
    "Energy_Rolling_Mean",
    "Energy_Rolling_Std"
]].head())