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

# Haftanın günü (0=Mon, 6=Sun): rutinleri yakalar
df["DayOfWeek"] = df["date"].dt.dayofweek

# Hafta sonu bilgisi: Hafta içi / hafta sonu tüketim farkını yakalar
df["Is_Weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

# Günün parçası: Sabah / Öğle / Akşam / Gece
# 05-11 Sabah, 12-16 Öğle, 17-21 Akşam, 22-04 Gece
def _part_of_day(hour: int) -> str:
    if 5 <= hour <= 11:
        return "Sabah"
    if 12 <= hour <= 16:
        return "Ogle"
    if 17 <= hour <= 21:
        return "Aksam"
    return "Gece"


df["Part_of_Day"] = df["Hour"].map(_part_of_day)
part_of_day_code_map = {"Gece": 0, "Sabah": 1, "Ogle": 2, "Aksam": 3}
df["Part_of_Day_Code"] = df["Part_of_Day"].map(part_of_day_code_map).astype(int)

# ----------------------------
# 2. SICAKLIK ÖZELLİKLERİ
# ----------------------------

temperature_columns = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
humidity_columns = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9"]

# Evin genel iç sıcaklığı
df["Average_Indoor_Temperature"] = df[temperature_columns].mean(axis=1)

# Evin genel iç nemi
df["Average_Indoor_Humidity"] = df[humidity_columns].mean(axis=1)

# İç ortam ile dış ortam sıcaklık farkı
df["Delta_T"] = df["Average_Indoor_Temperature"] - df["T_out"]

# Termal konfor endeksi (Discomfort Index benzeri)
# Formül (yaygın kullanım): DI = T - (0.55 - 0.0055 * RH) * (T - 14.5)
# RH yüzde (0-100) beklenir.
df["Thermal_Comfort_Index"] = (
    df["Average_Indoor_Temperature"]
    - (0.55 - 0.0055 * df["Average_Indoor_Humidity"]) * (df["Average_Indoor_Temperature"] - 14.5)
)

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
    "DayOfWeek",
    "Is_Weekend",
    "Part_of_Day",
    "Part_of_Day_Code",
    "Average_Indoor_Temperature",
    "Average_Indoor_Humidity",
    "Delta_T",
    "Thermal_Comfort_Index",
    "Energy_Lag_1",
    "Energy_Rolling_Mean",
    "Energy_Rolling_Std"
]].head())