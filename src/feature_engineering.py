import pandas as pd
import numpy as np

# Ham veriyi oku
df = pd.read_csv("data/raw/energydata_complete.csv")

# Tarih sütununu datetime formatına çevir
df["date"] = pd.to_datetime(df["date"])

# ----------------------------
# 1. ZAMAN ÖZELLİKLERİ
# ----------------------------

df["Hour"] = df["date"].dt.hour
df["DayOfWeek"] = df["date"].dt.dayofweek
df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

def part_of_day(hour):
    if 6 <= hour < 12:
        return "Sabah"
    elif 12 <= hour < 18:
        return "Öğle"
    elif 18 <= hour < 24:
        return "Akşam"
    else:
        return "Gece"

df["Part_of_Day"] = df["Hour"].apply(part_of_day)

# Çalışma saatleri: 09:00 - 18:00 arası
df["Is_Working_Hour"] = df["Hour"].between(9, 18).astype(int)

# ----------------------------
# 2. SICAKLIK VE NEM ÖZELLİKLERİ
# ----------------------------

temperature_columns = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
humidity_columns = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9"]

df["Average_Indoor_Temperature"] = df[temperature_columns].mean(axis=1)
df["Average_Indoor_Humidity"] = df[humidity_columns].mean(axis=1)

# İç ortam ile dış ortam sıcaklık farkı
df["Delta_T"] = df["Average_Indoor_Temperature"] - df["T_out"]

# Odalar arası sıcaklık farkı
df["Temp_Difference_Rooms"] = (
    df[temperature_columns].max(axis=1) - df[temperature_columns].min(axis=1)
)

# ----------------------------
# 3. TERMAL KONFOR ENDEKSİ
# ----------------------------

df["Thermal_Comfort_Index"] = (
    df["Average_Indoor_Temperature"]
    - 0.55
    * (1 - df["Average_Indoor_Humidity"] / 100)
    * (df["Average_Indoor_Temperature"] - 14.5)
)

# ----------------------------
# 4. GEÇMİŞ TÜKETİM ÖZELLİKLERİ
# ----------------------------

# Bir önceki ölçümdeki enerji tüketimi
df["Energy_Lag_1"] = df["Appliances"].shift(1)

# Son 6 ölçümün ortalaması
# Veri 10 dakikalık olduğu için 6 ölçüm yaklaşık 1 saate karşılık gelir
df["Energy_Rolling_Mean"] = df["Appliances"].rolling(window=6).mean()

# Son 6 ölçümdeki dalgalanma
df["Energy_Rolling_Std"] = df["Appliances"].rolling(window=6).std()

# Lag ve rolling işlemlerinden dolayı oluşan boş satırları temizle
df = df.dropna()

# ----------------------------
# 5. VERİYİ KAYDET
# ----------------------------

df.to_csv("data/processed/feature_engineered_data.csv", index=False)

print("Feature engineering tamamlandı.")
print("Yeni veri boyutu:", df.shape)

print(df[[
    "date",
    "Hour",
    "DayOfWeek",
    "Is_Weekend",
    "Part_of_Day",
    "Is_Working_Hour",
    "Average_Indoor_Temperature",
    "Average_Indoor_Humidity",
    "Delta_T",
    "Temp_Difference_Rooms",
    "Thermal_Comfort_Index",
    "Energy_Lag_1",
    "Energy_Rolling_Mean",
    "Energy_Rolling_Std"
]].head())