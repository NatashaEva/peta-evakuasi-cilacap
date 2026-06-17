"""
=======================================================
STEP 3 - ALOKASI JALUR DAN TEMPAT EVAKUASI BANJIR
Sistem Mitigasi Dinamis Terintegrasi - Kab. Cilacap
=======================================================
"""

import os, math
import pandas as pd
import numpy as np

# ── Path Data ─────────────────────────────────────────────────────────────
# ── Path Data (DIUBAH AGAR MENGARAH KE FOLDER DATA) ───────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
DATA      = os.path.join(BASE, "data")
PATH_PRED = os.path.join(DATA, "hasil_prediksi.csv")
PATH_SHEL = os.path.join(DATA, "tempatevakuasinew.csv") # ◄── Ubah BASE menjadi DATA di sini!
PATH_OUT  = os.path.join(DATA, "final_alokasi_evakuasi.csv")

print("=" * 60)
print("STEP 3: PROSES ALOKASI TEMPAT EVAKUASI TERDEKAT VIA CSV")
print("=" * 60)

# Validasi file input
if not os.path.exists(PATH_PRED):
    print(f"❌ Error: File {PATH_PRED} tidak ditemukan! Jalankan 'train_evaluate.py' dulu.")
    exit()
if not os.path.exists(PATH_SHEL):
    print(f"❌ Error: File {PATH_SHEL} tidak ditemukan! Pastikan file berada di folder utama.")
    exit()

# ── 1. LOAD DATA SHELTER DARI CSV (Tanpa Header) ──────────────────────────
# Karena file tidak memiliki baris nama kolom, kita baca manual indeksnya
df_shelter = pd.read_csv(PATH_SHEL, header=None)
print(f"✓ Berhasil memuat {len(df_shelter)} data titik evakuasi dari CSV milikmu.")

# ── 2. FUNGSI HAVERSINE (Hitung Jarak Geografis Asli) ─────────────────────
def hitung_jarak_km(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius bumi dalam Kilometer
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ── 3. LOAD DATA HASIL PREDIKSI RANDOM FOREST ────────────────────────────
df_desa = pd.read_csv(PATH_PRED)
print(f"✓ Berhasil memuat {len(df_desa)} data prediksi desa.")

# ── 4. PROSES PENCARIAN ALOKASI SHELTER TERDEKAT ─────────────────────────
alokasi_list = []

for _, row_desa in df_desa.iterrows():
    lat_d = float(row_desa['Latitude'])
    lon_d = float(row_desa['Longitude'])
    
    jarak_terpendek = float('inf')
    shelter_terpilih = "Tidak Ditemukan"
    
    # Looping mencari baris shelter terdekat berdasarkan indeks kolom CSV asli
    for _, row_shelter in df_shelter.iterrows():
        try:
            # Kolom 1 = Nama Tempat, Kolom 3 = Latitude, Kolom 4 = Longitude
            nama_s = row_shelter[1]
            lat_s  = float(row_shelter[3])
            lon_s  = float(row_shelter[4])
            
            jarak = hitung_jarak_km(lat_d, lon_d, lat_s, lon_s)
            if jarak < jarak_terpendek:
                jarak_terpendek = jarak
                shelter_terpilih = nama_s
        except Exception:
            continue # Skip jika ada baris kosong atau gagal diconvert ke float
            
    # Tentukan respon tindakan berdasarkan kelas hasil prediksi AI
    kelas_risiko = row_desa['kelas']
    if kelas_risiko == 'Kritis':
        tindakan = "EVAKUASI SEGERA (Jalur Merah Utama dibuka)"
    elif kelas_risiko == 'Tinggi':
        tindakan = "SIAGA (Mulai Mobilisasi Lansia & Anak-Anak)"
    elif kelas_risiko == 'Sedang':
        tindakan = "WASPADA (Pantau Saluran Air & Siapkan Tas Darurat)"
    else:
        tindakan = "AMAN (Tetap Monitor Info BMKG)"

    alokasi_list.append({
        'tempat_evakuasi_tujuan': str(shelter_terpilih).strip(),
        'jarak_ke_shelter_km': round(jarak_terpendek, 2),
        'manajemen_tindakan': tindakan
    })

df_alokasi = pd.DataFrame(alokasi_list)
# Gabungkan dengan dataframe utama desamu
df_final = pd.concat([df_desa, df_alokasi], axis=1)

# ── 5. SIMPAN HASIL AKHIR KE CSV FINAL ───────────────────────────────────
df_final.to_csv(PATH_OUT, index=False)
print("\n" + "─" * 60)
print("EVALUASI SIMULASI ALOKASI JALUR EVAKUASI:")
print("─" * 60)

summary_shelter = df_final.groupby('tempat_evakuasi_tujuan').size().reset_index(name='Jumlah Desa')
print(f"Top 5 Tempat Evakuasi Terpadat Hari Ini:")
print(summary_shelter.sort_values('Jumlah Desa', ascending=False).head(5).to_string(index=False))

print(f"\n✓ SELESAI LENGKAP! Data final disimpan di : {PATH_OUT}")
print("=" * 60)