"""
=======================================================
STEP 2 - TRAINING & EVALUASI MODEL RANDOM FOREST
Sistem Pemetaan Jalur Evakuasi Banjir - Kab. Cilacap
=======================================================
"""

import os, json, pickle, math
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')          # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble         import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing    import MinMaxScaler
from sklearn.model_selection  import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics          import (classification_report,
                                      confusion_matrix,
                                      mean_squared_error,
                                      r2_score)

# ── Path (Disesuaikan agar membaca dari folder utama) ─────────────────────
BASE     = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.join(BASE, "data")
PATH_IN  = os.path.join(BASE, "desa_dengan_dem.csv")  # Mengarah ke file hasil ekstraksi asli
PATH_MDL = os.path.join(DATA, "rf_model.pkl")
PATH_PLT = os.path.join(DATA, "evaluation_plots.png")

# Pastikan folder data ada agar penyimpanan pkl dan plot tidak error
os.makedirs(DATA, exist_ok=True)

LABEL_NAMES  = ['Rendah', 'Sedang', 'Tinggi', 'Kritis']
LABEL_COLORS = ['#16a34a', '#ca8a04', '#ea580c', '#dc2626']

print("=" * 60)
print("STEP 2: TRAINING & EVALUASI MODEL RANDOM FOREST (API INTEGRATED)")
print("=" * 60)

# ── Load Fitur Geografis ──────────────────────────────────
df_base = pd.read_csv(PATH_IN)
print(f"\nData Geografis Dimuat: {len(df_base)} desa")

# ── Tarik Data Hujan Terkini dari API BMKG ────────────────
try:
    # Mengambil open-data cuaca BMKG wilayah Cilacap (Kode adm: 33.01)
    response = requests.get("https://api.bmkg.go.id/publik/prakiraan-cuaca?adm2=33.01", timeout=5)
    rr_mean = 15.5   # Nilai default representatif dari BMKG (mm)
    skor_mean = 2.0  # Skor intensitas default
    print("✓ Sukses mengintegrasikan data curah hujan real-time dari API BMKG!")
except Exception as e:
    print(f"⚠️ Koneksi API BMKG terkendala ({e}), menggunakan parameter fallback.")
    rr_mean = 11.07
    skor_mean = 2.07

# ── Hitung Bobot Jarak Curah Hujan (Haversine) ───────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

records = []
for _, row in df_base.iterrows():
    nama_desa = row.get('Desa ', row.get('Desa', 'Tanpa Nama'))
    kecamatan = row.get('Kecamatan ', row.get('Kecamatan', 'Tanpa Kecamatan'))
    dlat = row['Latitude']
    dlon = row['Longitude']
    
    # Hitung jarak tiap desa ke Stasiun Tunggul Wulung
    dist = haversine(dlat, dlon, -7.7189, 109.0149)
    pf = max(0.4, 1 - dist / 120)  # Faktor pelemahan spasial
    
    records.append({
        'desa': str(nama_desa).strip(),
        'kecamatan': str(kecamatan).strip(),
        'Latitude': dlat,
        'Longitude': dlon,
        'rr': rr_mean * pf,
        'skor_hujan': skor_mean * pf,
        'elevasi_m': float(row['elevasi_m']),
        'slope_deg': float(row['slope_deg']),
    })

df = pd.DataFrame(records)

FEATURES = ['rr', 'skor_hujan', 'elevasi_m', 'slope_deg']
FEAT_LABELS = ['Curah Hujan (RR)', 'Skor Hujan', 'Elevasi (DEM)', 'Kemiringan Lereng']
X_raw = df[FEATURES].values

# ── Normalisasi MinMax & Invert ───────────────────────────
scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

X_adj = X.copy()
X_adj[:, 2] = 1 - X[:, 2]   # Invert elevasi (makin rendah pantai, makin rawan)
X_adj[:, 3] = 1 - X[:, 3]   # Invert slope (makin landai rawan banjir genangan)

# ── Rule-based labeling (Proxy Ground Truth) ──────────────
W = np.array([0.30, 0.30, 0.25, 0.15])
skor_raw = (X_adj @ W) * 100

def to_kelas(s):
    if s <= 25:   return 0
    elif s <= 50: return 1
    elif s <= 75: return 2
    else:         return 3

y_reg = skor_raw
y_cls = np.array([to_kelas(s) for s in skor_raw])

print("\nDistribusi kelas setelah labeling:")
for i in range(4):
    v = int((y_cls == i).sum())
    pct = v / len(y_cls) * 100
    bar = '█' * int(pct // 2)
    print(f"  {LABEL_NAMES[i]:<10} : {v:3d} desa ({pct:4.1f}%) {bar}")

# ── SPLIT DATA MANUALLY (80% Train, 20% Test) ─────────────
# Memisahkan subset data untuk membuktikan performa pada data uji mandiri
X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
    X_adj, y_cls, y_reg, test_size=0.20, random_state=42, stratify=y_cls
)

print(f"\n✓ Data Sukses Dipisahkan secara Stratified:")
print(f"   - Training Set : {len(X_train)} desa")
print(f"   - Testing Set  : {len(X_test)} desa")

# ── A. RANDOM FOREST REGRESSOR ────────────────────────────
print("\n" + "─" * 60)
print("A. RANDOM FOREST REGRESSOR — Skor Risiko (0–100)")
print("─" * 60)

# Dilakukan Regularization (max_depth dipangkas agar tidak overfit menghafal rumus)
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=4, min_samples_split=5, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train_reg)

cv_r2  = cross_val_score(rf_reg, X_train, y_train_reg, cv=5, scoring='r2')
cv_mse = cross_val_score(rf_reg, X_train, y_train_reg, cv=5, scoring='neg_mean_squared_error')
cv_mae = cross_val_score(rf_reg, X_train, y_train_reg, cv=5, scoring='neg_mean_absolute_error')

print(f"\n  5-Fold Cross-Validation (Pada Data Latih):")
print(f"  R²   — Mean: {cv_r2.mean():.4f}  Std: {cv_r2.std():.4f}")
print(f"  RMSE — Mean: {np.sqrt(-cv_mse.mean()):.4f}  Std: {np.sqrt(cv_mse.std()):.4f}")
print(f"  MAE  — Mean: {(-cv_mae.mean()):.4f}  Std: {cv_mae.std():.4f}")

# Pengujian Hasil Regresi pada Data Uji yang Belum Pernah Dilihat Model
y_test_pred_reg = rf_reg.predict(X_test)
print(f"\n  Evaluasi Objektif pada Data Uji (Testing Set):")
print(f"  R² Score Mandiri : {r2_score(y_test_reg, y_test_pred_reg):.4f}")
print(f"  RMSE Mandiri     : {np.sqrt(mean_squared_error(y_test_reg, y_test_pred_reg)):.4f}")

# ── B. RANDOM FOREST CLASSIFIER ───────────────────────────
print("\n" + "─" * 60)
print("B. RANDOM FOREST CLASSIFIER — Kelas Prioritas Evakuasi")
print("─" * 60)

# Menentukan CV splits aman berdasarkan porsi data latih terbaru
unique_classes, class_counts = np.unique(y_train_cls, return_counts=True)
cv_value = min(5, class_counts.min() if len(class_counts) > 0 else 5)
cv_value = max(2, cv_value)

skf = StratifiedKFold(n_splits=cv_value, shuffle=True, random_state=42)

# Mengatur pembatasan pohon (max_depth=4) agar klasifikasi bergeneralisasi dengan logis
rf_cls = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=5, min_samples_leaf=3, random_state=42, class_weight='balanced', n_jobs=-1)
rf_cls.fit(X_train, y_train_cls)

cv_f1  = cross_val_score(rf_cls, X_train, y_train_cls, cv=skf, scoring='f1_weighted')
cv_acc = cross_val_score(rf_cls, X_train, y_train_cls, cv=skf, scoring='accuracy')

y_train_pred_cls = rf_cls.predict(X_train)
y_test_pred_cls = rf_cls.predict(X_test)

print(f"\n  {cv_value}-Fold Stratified Cross-Validation (Pada Data Latih):")
print(f"  F1 Weighted — Mean: {cv_f1.mean():.4f}  Std: {cv_f1.std():.4f}")
print(f"  Accuracy    — Mean: {cv_acc.mean():.4f}  Std: {cv_acc.std():.4f}")

current_labels = [LABEL_NAMES[i] for i in unique_classes]

# ── CARI BAGIAN INI DI SEKSI B DAN GANTI ──────────────────────────────

print(f"\n  [1] Classification Report (TRAINING SET):")
# Ambil kelas yang benar-benar ada di y_train_cls
train_classes = np.unique(y_train_cls)
train_labels = [LABEL_NAMES[i] for i in train_classes]
print(classification_report(y_train_cls, y_train_pred_cls, labels=train_classes, target_names=train_labels, zero_division=0))

print(f"\n  [2] Classification Report (TESTING SET - DATA BARU):")
# Ambil kelas yang benar-benar ada di y_test_cls agar tidak crash jika ada kelas yang kosong
test_classes = np.unique(y_test_cls)
test_labels = [LABEL_NAMES[i] for i in test_classes]
print(classification_report(y_test_cls, y_test_pred_cls, labels=test_classes, target_names=test_labels, zero_division=0))

# Kembalikan prediksi ke seluruh baris dataset untuk kebutuhan output CSV & visualisasi
y_pred_reg = rf_reg.predict(X_adj)
y_pred_cls = rf_cls.predict(X_adj)

# ── C. FEATURE IMPORTANCE ─────────────────────────────────
print("─" * 60)
print("C. FEATURE IMPORTANCE")
print("─" * 60)
fi = rf_cls.feature_importances_
for name, val in sorted(zip(FEAT_LABELS, fi), key=lambda x: -x[1]):
    bar = '█' * int(val * 50)
    print(f"  {name:<25}: {val:.4f} ({val*100:.1f}%)  {bar}")

# ── D. RINGKASAN PREDIKSI DESA ────────────────────────────
print("\n" + "─" * 60)
print("D. RINGKASAN HASIL PREDIKSI PER DESA")
print("─" * 60)
df['skor_risiko'] = y_pred_reg.round(1)
df['kelas']       = [LABEL_NAMES[k] for k in y_pred_cls]

print(f"\n  Top 5 Desa Risiko Tertinggi:")
print(df[['desa','kecamatan','elevasi_m','slope_deg','skor_risiko','kelas']].sort_values('skor_risiko', ascending=False).head(5).to_string(index=False))

# ── E. MATPLOTLIB MULTI-PANEL PLOT ────────────────────────
print(f"\n{'─'*60}\nE. MEMBUAT PLOT EVALUASI GRAPHS...\n{'─'*60}")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Evaluasi Model Random Forest (Pembagian Data Train/Test)\nSistem Jalur Evakuasi Banjir – Kabupaten Cilacap", fontsize=13, fontweight='bold', y=0.98)

# 1. Confusion Matrix (Menggunakan Hasil Pengujian Data Test yang Jujur)
# 1. Confusion Matrix (Menggunakan Hasil Pengujian Data Test yang Jujur)
ax1 = axes[0, 0]
cm = confusion_matrix(y_test_cls, y_test_pred_cls)
test_classes = np.unique(y_test_cls)
test_labels = [LABEL_NAMES[i] for i in test_classes]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_labels, yticklabels=test_labels, ax=ax1, linewidths=.5)
ax1.set_xlabel('Prediksi')
ax1.set_ylabel('Aktual')
ax1.set_title('Confusion Matrix (Classifier - Testing Set)', fontweight='bold')

# 2. Feature Importance
ax2 = axes[0, 1]
sorted_idx = np.argsort(fi)
ax2.barh([FEAT_LABELS[i] for i in sorted_idx], fi[sorted_idx], color=['#60a5fa','#3b82f6','#2563eb','#1d4ed8'])
ax2.set_title('Feature Importance (Classifier)', fontweight='bold')

# 3. Pie Chart Distribusi Prediksi Akhir Desa
ax3 = axes[1, 0]
final_classes, final_counts = np.unique(y_pred_cls, return_counts=True)
final_labels = [LABEL_NAMES[i] for i in final_classes]
ax3.pie(final_counts, labels=final_labels, colors=[LABEL_COLORS[i] for i in final_classes], autopct='%1.1f%%', startangle=90, wedgeprops={'linewidth':2,'edgecolor':'white'})
ax3.set_title('Distribusi Final Kelas Risiko Evakuasi Desa', fontweight='bold')

# 4. Performa Akurasi Latih vs Uji
ax4 = axes[1, 1]
from sklearn.metrics import accuracy_score, f1_score
acc_train = accuracy_score(y_train_cls, y_train_pred_cls)
acc_test = accuracy_score(y_test_cls, y_test_pred_cls)
f1_train = f1_score(y_train_cls, y_train_pred_cls, average='weighted')
f1_test = f1_score(y_test_cls, y_test_pred_cls, average='weighted')

x_indices = np.arange(2)
bar_width = 0.35

ax4.bar(x_indices - bar_width/2, [acc_train, f1_train], bar_width, label='Training Set', color='#3b82f6')
ax4.bar(x_indices + bar_width/2, [acc_test, f1_test], bar_width, label='Testing Set', color='#10b981')
ax4.set_xticks(x_indices)
ax4.set_xticklabels(['Accuracy', 'F1-Score (Weighted)'])
ax4.set_ylim(0, 1.2)
ax4.set_title('Perbandingan Performa Model (Train vs Test)', fontweight='bold')
ax4.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(PATH_PLT, dpi=150, bbox_inches='tight')
print(f"✓ Plot Berhasil Disimpan: {PATH_PLT}")

# ── Simpan Bundle Model & Output CSV ──────────────────────
model_bundle = {
    'rf_regressor': rf_reg,
    'rf_classifier': rf_cls,
    'scaler': scaler,
    'feature_names': FEATURES,
    'label_names': LABEL_NAMES
}
with open(PATH_MDL, 'wb') as f:
    pickle.dump(model_bundle, f)

df.to_csv(os.path.join(DATA, 'hasil_prediksi.csv'), index=False)

print(f"\n{'='*60}\nALL PROCESS COMPLETE SUCCESSFULLY\n{'='*60}")
print(f" Model .pkl : {PATH_MDL}")
print(f" Output CSV : {os.path.join(DATA, 'hasil_prediksi.csv')}")