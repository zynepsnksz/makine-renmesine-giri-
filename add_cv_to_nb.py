import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Markdown Intro
md_intro = """# Faz 2: Gözetimli Model Karşılaştırması ve Çapraz Doğrulama (Cross Validation)
Gözetimsiz modellerin (Isolation Forest, vs.) performansının düşük kalması üzerine, **Gözetimli Öğrenme (Supervised Learning)** modellerine geçiş yaptık. 
Modelimizin sonuçlarının "şansa" iyi çıkmadığını hocaya kanıtlamak için, veriyi rastgele bir kez bölmek yerine **5-Katlı Çapraz Doğrulama (5-Fold Stratified Cross Validation)** kullanıyoruz. Bu yöntem, veriyi 5 farklı parçaya böler, 5 kez farklı verilerle eğitir ve ortalama başarıyı verir. Akademik projelerde altın standart budur!"""

# Cell 2: Setup
code_setup = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lazypredict.Supervised import LazyClassifier

try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False
    print("xgboost yüklü değil.")

plt.style.use("default")
sns.set_theme(style="white", rc={"axes.facecolor": "white", "figure.facecolor": "white"})
"""

# Cell 3: Load Data
code_load = """file_path = os.path.join('data', 'processed', 'final_results.csv')
df = pd.read_csv(file_path)

drop_cols = ['date', 'is_anomaly', 'z_score', 'z_score_abs', 'z_pred', 
             'iforest_pred', 'lof_pred', 'ocsvm_pred', 'final_pred', 
             'Part_of_Day', 'DayOfWeek']

df = df.dropna()
X = df.drop(columns=drop_cols)
y = df['is_anomaly']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Toplam Veri Boyutu: {X.shape}")"""

# Cell 4: Markdown LazyPredict
md_lazy = """## 1. LazyPredict ile Genel Tarama
LazyPredict arka planda basit train_test_split kullanır. Bu aşamayı dünyadaki algoritmaların genel bir "Röntgenini" çekmek için kullanıyoruz."""

# Cell 5: Code LazyPredict
code_lazy = """X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
print("LazyPredict çalışıyor (1-2 dakika sürebilir)...")
models_summary, _ = clf.fit(X_train, X_test, y_train, y_test)

display(models_summary.sort_values(by="F1 Score", ascending=False).head(10))"""

# Cell 6: Markdown CV
md_cv = """## 2. Şampiyonların 5-Katlı Çapraz Doğrulaması (Cross-Validation)
LazyPredict bize yol gösterdi. Şimdi en mantıklı ve güçlü modelleri (Random Forest, XGBoost ve referans için Logistic Regresyon) alıp, **5-Fold Stratified Cross Validation** testine sokuyoruz. Bu test, modelin ezber yapmadığını (overfitting olmadığını) kanıtlar."""

# Cell 7: Code CV
code_cv = """models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}
if xgb_available:
    models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

cv_results_list = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("5-Fold Cross Validation testleri yapılıyor...")
for name, model in models.items():
    # cross_validate ile Precision, Recall ve F1 skorlarını 5 farklı parça için hesaplıyoruz
    scoring = ['precision', 'recall', 'f1']
    scores = cross_validate(model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    cv_results_list.append({
        "Model": name,
        "F1-Score (Ortalama)": scores['test_f1'].mean(),
        "F1-Score (Sapma)": scores['test_f1'].std(),
        "Precision (Ortalama)": scores['test_precision'].mean(),
        "Recall (Ortalama)": scores['test_recall'].mean()
    })

cv_df = pd.DataFrame(cv_results_list).sort_values(by="F1-Score (Ortalama)", ascending=False)
display(cv_df)"""

# Cell 8: Code Visual CV
code_vis = """plt.figure(figsize=(10, 6))
# Hata çubukları (error bars) ekleyerek sapmayı gösteriyoruz
ax = sns.barplot(x="F1-Score (Ortalama)", y="Model", data=cv_df, palette="mako", 
                 xerr=cv_df["F1-Score (Sapma)"], capsize=.2)

plt.title("5-Fold Cross Validation F1-Skoru Karşılaştırması", pad=15, fontsize=14, fontweight='bold')
plt.xlabel("Ortalama F1-Skoru", labelpad=10)
plt.ylabel("")

for i, v in enumerate(cv_df["F1-Score (Ortalama)"]):
    ax.text(v + 0.01, i, f"%{v*100:.1f}", va='center', fontweight='bold', color='black')

plt.xlim(0, 1.1)
plt.tight_layout()
plt.show()"""

# Add cells
nb['cells'] = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_code_cell(code_setup),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell(md_lazy),
    nbf.v4.new_code_cell(code_lazy),
    nbf.v4.new_markdown_cell(md_cv),
    nbf.v4.new_code_cell(code_cv),
    nbf.v4.new_code_cell(code_vis)
]

with open('Faz2_Supervised_Model_Karsilastirma.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Cross Validation başarıyla notebook'a eklendi!")
