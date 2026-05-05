import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# Veri setini yükle
file_path = 'energydata_complete.csv'
df = pd.read_csv(file_path)

# Tarih sütununu datetime formatına çevir
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Sadece Ocak ayından bir haftalık veri alalım (görselleştirme kolay olsun diye)
df_subset = df.loc['2016-01-12':'2016-01-18'].copy()

# Anomali etiketini tutacak yeni bir sütun ekleyelim (0: Normal, 1: Anomali)
df_subset['is_anomaly'] = 0
df_subset['Appliances_Original'] = df_subset['Appliances'].copy()

# --- SENTETİK ANOMALİ ENJEKSİYONU ---
# Senaryo 1: Tost Makinesi Açık Unutuldu (Örn: Gece 02:00 sularında)
# Gece düşük tüketim olan bir zamanı seçelim
anomaly_start_1 = pd.to_datetime('2016-01-14 02:00:00')
# Veri 10 dakikalık periyotlar halinde. Tost makinesi 40 dakika açık kalsın (4 periyot)
periods_1 = 4 
extra_watt_1 = 800 # Tost makinesi 800 Wh

for i in range(periods_1):
    current_time = anomaly_start_1 + pd.Timedelta(minutes=10*i)
    if current_time in df_subset.index:
        df_subset.loc[current_time, 'Appliances'] += extra_watt_1
        df_subset.loc[current_time, 'is_anomaly'] = 1

# Senaryo 2: Kombi/Isıtıcı Arızası veya Kapı Açık Kaldı (Gündüz yüksek tüketim)
anomaly_start_2 = pd.to_datetime('2016-01-16 14:30:00')
periods_2 = 6 # 1 saat boyunca
extra_watt_2 = 600

for i in range(periods_2):
    current_time = anomaly_start_2 + pd.Timedelta(minutes=10*i)
    if current_time in df_subset.index:
        df_subset.loc[current_time, 'Appliances'] += extra_watt_2
        df_subset.loc[current_time, 'is_anomaly'] = 1

# --- GÖRSELLEŞTİRME ---
plt.figure(figsize=(15, 6))
plt.plot(df_subset.index, df_subset['Appliances_Original'], label='Orijinal Tüketim (Normal)', color='blue', alpha=0.6)
plt.plot(df_subset.index, df_subset['Appliances'], label='Anomalili Tüketim', color='red', alpha=0.5, linestyle='--')

# Anomalileri işaretle
anomalies = df_subset[df_subset['is_anomaly'] == 1]
plt.scatter(anomalies.index, anomalies['Appliances'], color='darkred', s=50, label='Açık Unutulan Cihaz (Anomali)', zorder=5)

plt.title('Sentetik Anomali Enjeksiyonu: Açık Unutulan Cihaz Senaryosu', fontsize=14, fontweight='bold')
plt.xlabel('Tarih', fontsize=12)
plt.ylabel('Enerji Tüketimi (Wh)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

# Grafiği kaydet
plt.savefig('anomaly_visualization.png', dpi=300)
print("Anomali enjeksiyonu tamamlandı ve grafik 'anomaly_visualization.png' olarak kaydedildi.")
print("Ayrıca veri setine 'is_anomaly' adında hedef (target) değişken eklendi.")

# Anomalili veri setinin küçük bir kısmını kaydet
df_subset.to_csv('energydata_with_anomalies.csv')
