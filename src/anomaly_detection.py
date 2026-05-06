import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

# ----------------------------
# 1. KLASÖR
# ----------------------------
os.makedirs("outputs", exist_ok=True)

# ----------------------------
# 2. VERİYİ OKU
# ----------------------------
df = pd.read_csv("data/processed/anomaly_dataset_clean.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# ----------------------------
# 3. FEATURES
# ----------------------------
features = [
    "Appliances",
    "Hour",
    "Is_Weekend",
    "Average_Indoor_Temperature",
    "Delta_T",
    "Energy_Lag_1",
    "Energy_Rolling_Mean",
    "Energy_Rolling_Std"
]

X = df[features]
y_true = df["is_anomaly"]

# ----------------------------
# 4. ÖLÇEKLEME
# ----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Gerçek anomali oranı
contamination_rate = y_true.mean()

print("\nGerçek anomali oranı:", contamination_rate)

# ----------------------------
# 5. Z-SCORE ALGORİTMASI
# ----------------------------
df["z_score"] = df.groupby("Hour")["Appliances"].transform(
    lambda x: (x - x.mean()) / x.std()
)

df["z_score_abs"] = df["z_score"].abs()

threshold_list = [0.97, 0.975, 0.98, 0.985, 0.99]

best_z_f1 = 0
best_z_threshold = None
best_z_pred = None

for t in threshold_list:
    threshold = df["z_score_abs"].quantile(t)
    pred = (df["z_score_abs"] > threshold).astype(int)

    f1 = f1_score(y_true, pred)

    if f1 > best_z_f1:
        best_z_f1 = f1
        best_z_threshold = threshold
        best_z_pred = pred

df["z_pred"] = best_z_pred

print("\nZ-score en iyi threshold:", best_z_threshold)

# ----------------------------
# 6. ISOLATION FOREST
# ----------------------------
iforest = IsolationForest(
    n_estimators=200,
    contamination=contamination_rate,
    random_state=42
)

iforest_raw = iforest.fit_predict(X_scaled)

df["iforest_pred"] = pd.Series(iforest_raw).map({
    1: 0,
    -1: 1
}).values

# ----------------------------
# 7. LOCAL OUTLIER FACTOR
# ----------------------------
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=contamination_rate
)

lof_raw = lof.fit_predict(X_scaled)

df["lof_pred"] = pd.Series(lof_raw).map({
    1: 0,
    -1: 1
}).values

# ----------------------------
# 8. ONE-CLASS SVM
# ----------------------------
ocsvm = OneClassSVM(
    kernel="rbf",
    nu=contamination_rate,
    gamma="scale"
)

ocsvm_raw = ocsvm.fit_predict(X_scaled)

df["ocsvm_pred"] = pd.Series(ocsvm_raw).map({
    1: 0,
    -1: 1
}).values

# ----------------------------
# 9. ALGORİTMALARI KARŞILAŞTIR
# ----------------------------
models = {
    "Z-score": df["z_pred"],
    "Isolation Forest": df["iforest_pred"],
    "Local Outlier Factor": df["lof_pred"],
    "One-Class SVM": df["ocsvm_pred"]
}

results = []

print("\n===== ALGORİTMA KARŞILAŞTIRMA =====\n")

for model_name, y_pred in models.items():
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    results.append({
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    })

    print(f"\n--- {model_name} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, zero_division=0))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("f1_score", ascending=False)

print("\n===== ÖZET SONUÇLAR =====")
print(results_df)

# ----------------------------
# 10. EN İYİ ALGORİTMAYI SEÇ
# ----------------------------
best_model_name = results_df.iloc[0]["model"]

print("\nEn iyi algoritma:", best_model_name)

if best_model_name == "Z-score":
    df["final_pred"] = df["z_pred"]
elif best_model_name == "Isolation Forest":
    df["final_pred"] = df["iforest_pred"]
elif best_model_name == "Local Outlier Factor":
    df["final_pred"] = df["lof_pred"]
elif best_model_name == "One-Class SVM":
    df["final_pred"] = df["ocsvm_pred"]

# ----------------------------
# 11. FINAL RAPOR
# ----------------------------
print("\n===== FINAL MODEL RAPORU =====")
print("Seçilen algoritma:", best_model_name)
print(classification_report(y_true, df["final_pred"], zero_division=0))

# ----------------------------
# 12. GRAFİK
# ----------------------------
plt.figure(figsize=(15, 5))

plt.plot(
    df["date"],
    df["Appliances"],
    label="Energy",
    alpha=0.7
)

anomalies = df[df["final_pred"] == 1]

plt.scatter(
    anomalies["date"],
    anomalies["Appliances"],
    label=f"Anomaly - {best_model_name}",
    s=35
)

plt.legend()
plt.title(f"Final Anomaly Detection - Best Model: {best_model_name}")
plt.xlabel("Date")
plt.ylabel("Energy Consumption")
plt.grid(True, linestyle=":", alpha=0.7)
plt.tight_layout()

plt.savefig("outputs/final_anomaly_detection.png", dpi=300)
plt.close()

# ----------------------------
# 13. SONUÇLARI KAYDET
# ----------------------------
df.to_csv("data/processed/final_results.csv", index=False)
results_df.to_csv("data/processed/model_comparison_results.csv", index=False)

print("\nKaydedildi:")
print("data/processed/final_results.csv")
print("data/processed/model_comparison_results.csv")
print("outputs/final_anomaly_detection.png")