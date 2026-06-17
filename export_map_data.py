"""
=======================================================
STEP 3 - EXPORT DATA UNTUK PETA (index.html)
Sistem Pemetaan Jalur Evakuasi Banjir - Kab. Cilacap
=======================================================
Jalankan SETELAH 2_train_evaluate.py

Membaca model yang sudah dilatih dan menghasilkan
data_peta.json yang digunakan oleh index.html
"""

import os, json, math, pickle
import numpy as np
import pandas as pd

BASE     = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.join(BASE, "data")

PATH_MDL   = os.path.join(DATA, "rf_model.pkl")
PATH_FEAT  = os.path.join(DATA, "features.csv")
PATH_EVAK  = os.path.join(DATA, "tempatevakuasinew.csv")
PATH_JALAN = os.path.join(DATA, "jaringan_jalan_cilacap.json")
PATH_OUT   = os.path.join(BASE, "data_peta.json")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

print("=" * 55)
print("STEP 3: EXPORT DATA PETA")
print("=" * 55)

# ── Load model ────────────────────────────────────────────
print("\n[1/4] Memuat model RF...")
with open(PATH_MDL, 'rb') as f:
    bundle = pickle.load(f)
rf_reg  = bundle['rf_regressor']
rf_cls  = bundle['rf_classifier']
scaler  = bundle['scaler']
LABELS  = bundle['label_names']
COLORS  = ['#16a34a', '#ca8a04', '#ea580c', '#dc2626']
print(f"    → Model dimuat: R²={bundle['cv_scores']['r2_mean']}, "
      f"F1={bundle['cv_scores']['f1_mean']}")

# ── Load & prediksi ulang ─────────────────────────────────
print("\n[2/4] Menjalankan prediksi RF untuk semua desa...")
df = pd.read_csv(PATH_FEAT)
X_raw = df[['rr','skor_hujan','elevasi_m','slope_deg']].values
X     = scaler.transform(X_raw)
X_adj = X.copy()
X_adj[:, 2] = 1 - X[:, 2]
X_adj[:, 3] = 1 - X[:, 3]

pred_skor  = rf_reg.predict(X_adj)
pred_kelas = rf_cls.predict(X_adj)
fi         = rf_cls.feature_importances_

# ── Load evakuasi ─────────────────────────────────────────
df_evak = pd.read_csv(PATH_EVAK, header=None,
    names=['no','nama','alamat','lat','lon','kode1',
           'kapasitas','luas','lantai','kode_desa',
           'kode_kec','gmaps','jenis'])
df_evak['lat'] = pd.to_numeric(df_evak['lat'], errors='coerce')
df_evak['lon'] = pd.to_numeric(df_evak['lon'], errors='coerce')
df_evak = df_evak.dropna(subset=['lat','lon'])
df_evak = df_evak[(df_evak['lon'] > 107) & (df_evak['lon'] < 111)]
evakuasi = df_evak[['nama','alamat','lat','lon',
                     'kapasitas','jenis']].to_dict('records')

# ── Build desa_rawan & jalur ──────────────────────────────
desa_rawan, jalur = [], []
for i, row in df.iterrows():
    ki   = int(pred_kelas[i])
    clr  = COLORS[ki]
    best = min(evakuasi,
               key=lambda e: haversine(row['lat'], row['lon'],
                                       e['lat'], e['lon']))
    dist = haversine(row['lat'], row['lon'],
                     best['lat'], best['lon'])
    skor = round(float(pred_skor[i]), 1)

    desa_rawan.append({
        'desa':        row['desa'],
        'kecamatan':   row['kecamatan'],
        'lat':         row['lat'],
        'lon':         row['lon'],
        'skor_risiko': skor,
        'kelas':       LABELS[ki],
        'color':       clr,
        'rr':          round(row['rr'], 2),
        'skor_hujan':  round(row['skor_hujan'], 2),
        'elevasi_m':   round(row['elevasi_m'], 1),
        'slope_deg':   round(row['slope_deg'], 2),
    })
    jalur.append({
        'from': {'lat':   row['lat'],
                 'lon':   row['lon'],
                 'name':  row['desa'],
                 'kec':   row['kecamatan']},
        'to':   {'lat':   best['lat'],
                 'lon':   best['lon'],
                 'name':  best['nama']},
        'dist_km':     round(dist, 2),
        'skor_risiko': skor,
        'kelas':       LABELS[ki],
        'color':       clr,
        'rr':          round(row['rr'], 2),
        'elevasi_m':   round(row['elevasi_m'], 1),
        'slope_deg':   round(row['slope_deg'], 2),
    })

# ── Load jalan ────────────────────────────────────────────
print("\n[3/4] Memuat jaringan jalan...")
with open(PATH_JALAN) as f:
    gj = json.load(f)
SKIP = {'residential', 'unclassified', 'living_street'}
roads = []
for feat in gj['features']:
    hw = feat['properties'].get('highway', '')
    if isinstance(hw, list): hw = hw[0]
    if hw in SKIP: continue
    coords = [[c[1], c[0]] for c in feat['geometry']['coordinates']]
    roads.append({'highway': hw,
                  'name':    feat['properties'].get('name') or '',
                  'coords':  coords})
print(f"    → {len(roads)} segmen jalan dimuat")

# ── Feature importance ────────────────────────────────────
feat_importance = [
    {'name':  n, 'value': round(float(v), 4),
     'pct':   round(float(v)*100, 1)}
    for n, v in zip(
        ['Curah Hujan (RR)', 'Skor Hujan',
         'Elevasi (DEM)', 'Kemiringan Lereng'], fi)
]

# ── Simpan JSON ───────────────────────────────────────────
print("\n[4/4] Menyimpan data_peta.json...")
out = {
    'evakuasi':        evakuasi,
    'desa_rawan':      desa_rawan,
    'jalan':           roads,
    'jalur':           jalur,
    'feat_importance': feat_importance,
    'model_info': {
        'r2_cv':    bundle['cv_scores']['r2_mean'],
        'f1_cv':    bundle['cv_scores']['f1_mean'],
        'acc_cv':   bundle['cv_scores']['accuracy_mean'],
        'n_trees':  200,
        'n_features': 4,
        'features': ['Curah Hujan (RR)', 'Skor Hujan',
                     'Elevasi (DEM)', 'Kemiringan Lereng'],
        'bobot':    {'rr': 0.30, 'skor_hujan': 0.30,
                     'elevasi': 0.25, 'slope': 0.15}
    }
}
with open(PATH_OUT, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False)

size_mb = os.path.getsize(PATH_OUT) / 1024 / 1024
print(f"\n{'='*55}")
print("EXPORT SELESAI")
print(f"{'='*55}")
print(f"  Output : {PATH_OUT}")
print(f"  Size   : {size_mb:.2f} MB")
print(f"\n  Distribusi kelas jalur evakuasi:")
from collections import Counter
kelas_count = Counter(j['kelas'] for j in jalur)
for k in ['Rendah','Sedang','Tinggi','Kritis']:
    print(f"    {k:<10}: {kelas_count.get(k,0)} jalur")
print(f"\n  Sekarang buka index.html di browser!")