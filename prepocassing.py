"""
=======================================================
STEP 1 - PREPROCESSING DATA
Sistem Pemetaan Jalur Evakuasi Banjir - Kab. Cilacap
=======================================================
Menjalankan file ini akan:
1. Load semua dataset (evakuasi, desa rawan, curah hujan, DEM, jalan)
2. Ekstraksi fitur elevasi & slope dari DEM per desa
3. Hitung fitur curah hujan per desa
4. Simpan hasil ke data/features.csv untuk training
"""

import json, math, re, os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol

# ── Path data (sesuaikan jika perlu) ─────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")

PATH_EVAKUASI  = os.path.join(DATA, "tempatevakuasinew.csv")
PATH_DESA      = os.path.join(DATA, "Data_Desa_Rawan_Banjir_di_Cilacap.xlsx")
PATH_HUJAN     = os.path.join(DATA, "converted__1_.geojson")
PATH_DEM       = os.path.join(DATA, "dem_cilacap.tif")
PATH_JALAN     = os.path.join(DATA, "jaringan_jalan_cilacap.json")
PATH_OUT       = os.path.join(DATA, "features.csv")

# ── Haversine ─────────────────────────────────────────────
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# ─────────────────────────────────────────────────────────
print("=" * 55)
print("STEP 1: PREPROCESSING DATA")
print("=" * 55)

# ── 1. Load Titik Evakuasi ────────────────────────────────
print("\n[1/5] Memuat data titik evakuasi...")
df_evak = pd.read_csv(PATH_EVAKUASI, header=None,
    names=['no','nama','alamat','lat','lon','kode1',
           'kapasitas','luas','lantai','kode_desa',
           'kode_kec','gmaps','jenis'])
df_evak['lat'] = pd.to_numeric(df_evak['lat'], errors='coerce')
df_evak['lon'] = pd.to_numeric(df_evak['lon'], errors='coerce')
df_evak = df_evak.dropna(subset=['lat','lon'])
df_evak = df_evak[(df_evak['lon'] > 107) & (df_evak['lon'] < 111)]
evakuasi = df_evak[['nama','alamat','lat','lon',
                     'kapasitas','jenis']].to_dict('records')
print(f"    → {len(evakuasi)} titik evakuasi valid")

# ── 2. Load Desa Rawan Banjir ─────────────────────────────
print("\n[2/5] Memuat data desa rawan banjir...")
df_desa = pd.read_excel(PATH_DESA, header=3)
df_desa['Latitude']   = pd.to_numeric(df_desa['Latitude'],  errors='coerce')
df_desa['Longitude']  = pd.to_numeric(df_desa['Longitude'], errors='coerce')
df_desa['Kecamatan '] = df_desa['Kecamatan '].ffill()
df_desa = df_desa.dropna(subset=['Latitude','Longitude'])
df_desa['Desa ']      = df_desa['Desa '].astype(str).str.strip()
df_desa['Kecamatan '] = df_desa['Kecamatan '].astype(str).str.strip()
print(f"    → {len(df_desa)} desa rawan banjir")

# ── 3. Load Curah Hujan ───────────────────────────────────
print("\n[3/5] Memuat data curah hujan...")
with open(PATH_HUJAN) as f:
    gj_h = json.load(f)
hujan_list = [{'tanggal': feat['properties']['Tanggal'],
               'rr':      float(feat['properties']['RR']),
               'skor':    int(feat['properties']['skor_hujan'])}
              for feat in gj_h['features']]
df_hujan = pd.DataFrame(hujan_list)
rr_mean_global   = df_hujan['rr'].mean()
skor_mean_global = df_hujan['skor'].mean()
rr_max_global    = df_hujan['rr'].max()
print(f"    → {len(df_hujan)} record curah hujan")
print(f"    → RR rata-rata: {rr_mean_global:.2f} mm/hari")
print(f"    → RR maksimum : {rr_max_global:.2f} mm/hari")
print(f"    → Skor rata-rata: {skor_mean_global:.2f}")

# ── 4. Ekstraksi Elevasi & Slope dari DEM ─────────────────
print("\n[4/5] Mengekstrak elevasi & slope dari DEM SRTM 30m...")
src        = rasterio.open(PATH_DEM)
dem_data   = src.read(1).astype(float)
nodata     = src.nodata or -32768
dem_data[dem_data == nodata] = np.nan
transform  = src.transform
res_m      = src.res[0] * 111000   # derajat → meter

dz_dy, dz_dx = np.gradient(dem_data, res_m, res_m)
slope_deg    = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

print(f"    → DEM shape  : {dem_data.shape}")
print(f"    → Elevasi    : {np.nanmin(dem_data):.1f} – {np.nanmax(dem_data):.1f} m")
print(f"    → Slope      : {np.nanmin(slope_deg):.2f}° – {np.nanmax(slope_deg):.2f}°")

RADIUS_PX = 5
elev_list, slope_list = [], []
for _, row in df_desa.iterrows():
    try:
        r, c   = rowcol(transform, row['Longitude'], row['Latitude'])
        r0, r1 = max(0, r-RADIUS_PX), min(dem_data.shape[0], r+RADIUS_PX)
        c0, c1 = max(0, c-RADIUS_PX), min(dem_data.shape[1], c+RADIUS_PX)
        elev_list.append(float(np.nanmean(dem_data[r0:r1, c0:c1])))
        slope_list.append(float(np.nanmean(slope_deg[r0:r1, c0:c1])))
    except:
        elev_list.append(np.nan)
        slope_list.append(np.nan)
src.close()
df_desa['elevasi_m'] = elev_list
df_desa['slope_deg'] = slope_list
df_desa = df_desa.dropna(subset=['elevasi_m','slope_deg'])
print(f"    → Berhasil diekstrak: {len(df_desa)} desa")

# ── 5. Hitung Semua Fitur per Desa ───────────────────────
print("\n[5/5] Menghitung fitur per desa...")
KOORDINAT_STASIUN = (-7.7189, 109.0149)
records = []
for _, row in df_desa.iterrows():
    dlat = row['Latitude']
    dlon = row['Longitude']

    # Fitur curah hujan (proximity-weighted dari stasiun)
    dist_st  = haversine(dlat, dlon, *KOORDINAT_STASIUN)
    pf       = max(0.4, 1 - dist_st / 120)
    rr_desa  = rr_mean_global * pf
    sk_desa  = skor_mean_global * pf

    # Jarak ke titik evakuasi terdekat (Nearest Neighbor)
    best_dist = min(haversine(dlat, dlon, e['lat'], e['lon'])
                    for e in evakuasi)
    best_evak = min(evakuasi,
                    key=lambda e: haversine(dlat, dlon, e['lat'], e['lon']))

    records.append({
        'desa':             row['Desa '],
        'kecamatan':        row['Kecamatan '],
        'lat':              dlat,
        'lon':              dlon,
        'rr':               round(rr_desa, 4),
        'skor_hujan':       round(sk_desa, 4),
        'elevasi_m':        round(row['elevasi_m'], 2),
        'slope_deg':        round(row['slope_deg'], 4),
        'jarak_evakuasi_km':round(best_dist, 4),
        'nama_evakuasi':    best_evak['nama'],
        'lat_evakuasi':     best_evak['lat'],
        'lon_evakuasi':     best_evak['lon'],
    })

df_feat = pd.DataFrame(records)
df_feat.to_csv(PATH_OUT, index=False)

print(f"\n{'=' * 55}")
print(f"PREPROCESSING SELESAI")
print(f"{'=' * 55}")
print(f"Total desa diproses : {len(df_feat)}")
print(f"Output tersimpan    : {PATH_OUT}")
print(f"\nSample data fitur:")
print(df_feat[['desa','rr','skor_hujan','elevasi_m',
               'slope_deg','jarak_evakuasi_km']].head(8).to_string())
print(f"\nStatistik fitur:")
print(df_feat[['rr','skor_hujan','elevasi_m',
               'slope_deg','jarak_evakuasi_km']].describe().round(3).to_string())