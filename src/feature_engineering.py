import pandas as pd
import numpy as np

df = pd.read_csv("data/raw/energydata_complete.csv")

df["date"] = pd.to_datetime(df["date"])

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

temperature_columns = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
humidity_columns = ["RH_1", "RH_2", "RH_3", "RH_4", "RH_5", "RH_6", "RH_7", "RH_8", "RH_9"]

df["Average_Indoor_Temperature"] = df[temperature_columns].mean(axis=1)
df["Average_Indoor_Humidity"] = df[humidity_columns].mean(axis=1)

df["Delta_T"] = df["Average_Indoor_Temperature"] - df["T_out"]

df["Thermal_Comfort_Index"] = (
    df["Average_Indoor_Temperature"]
    - 0.55 * (1 - df["Average_Indoor_Humidity"] / 100)
    * (df["Average_Indoor_Temperature"] - 14.5)
)

df.to_csv("data/processed/feature_engineered_data.csv", index=False)

print("Feature engineering tamamlandı.")
print(df[[
    "date",
    "Hour",
    "DayOfWeek",
    "Is_Weekend",
    "Part_of_Day",
    "Average_Indoor_Temperature",
    "Average_Indoor_Humidity",
    "Delta_T",
    "Thermal_Comfort_Index"
]].head())