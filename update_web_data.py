"""
=======================================================
STEP 4 - UPDATE DATA WEB INTERAKTIF (FALLBACK NEAREST)
Memaksa Rute Evakuasi Menuju ke Posko Terdekat
=======================================================
"""

import os
import json
import math
import pandas as pd

# ── Path Jalur File ───────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
PATH_FINAL_CSV = os.path.join(DATA_DIR, "final_alokasi_evakuasi.csv")
PATH_SHELTER_CSV = os.path.join(DATA_DIR, "tempatevakuasinew.csv")
PATH_INDEX_HTML = os.path.join(BASE, "index.html")

print("=" * 60)
print("FORCE ALLOCATION: MENGARAHKAN RUTE KE POSKO TERDEKAT")
print("=" * 60)

if not os.path.exists(PATH_FINAL_CSV):
    print(f"❌ Error: File {PATH_FINAL_CSV} tidak ditemukan!")
    exit()
if not os.path.exists(PATH_SHELTER_CSV):
    print(f"❌ Error: File {PATH_SHELTER_CSV} tidak ditemukan!")
    exit()
if not os.path.exists(PATH_INDEX_HTML):
    print(f"❌ Error: File {PATH_INDEX_HTML} tidak ditemukan!")
    exit()

# Fungsi hitung jarak koordinat lurus (Haversine Formula) dalam KM
def hitung_jarak(lat1, lon1, lat2, lon2):
    R = 6371.0 # Radius bumi dalam km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ── 1. READ DATA ALL SHELTERS ─────────────────────────────────────────────
shelter_list = []
shelter_dict = {}
df_shelter = pd.read_csv(PATH_SHELTER_CSV, header=None)

for _, r in df_shelter.iterrows():
    try:
        nama_sh = str(r[1]).strip()
        raw_kapasitas = str(r[6]).strip()
        if raw_kapasitas == "\\N" or raw_kapasitas == "" or not raw_kapasitas.isdigit():
            raw_kapasitas = str(r[5]).strip()
        kapasitas_angka = int(raw_kapasitas) if raw_kapasitas.isdigit() else 0
        
        sh_obj = {
            "nama": nama_sh,
            "alamat": str(r[2]).strip(),
            "lat": float(r[3]),
            "lon": float(r[4]),
            "kapasitas": kapasitas_angka,
            "jenis": str(r[12]).strip() if pd.notna(r[12]) else "Gedung Evakuasi"
        }
        shelter_list.append(sh_obj)
        shelter_dict[nama_sh] = sh_obj
    except:
        continue

# ── 2. READ DATA DESA & RE-ROUTE KE TERDEKAT ──────────────────────────────
df_desa = pd.read_csv(PATH_FINAL_CSV)

list_desa_rawan = []
list_jalur = []

for _, r in df_desa.iterrows():
    nama_desa = "Tanpa Nama"
    for opsi_kolom in ['Desa', 'nama_desa', 'desa', 'Desa_Rawan']:
        if opsi_kolom in r:
            nama_desa = str(r[opsi_kolom]).strip()
            break
            
    kecamatan = str(r.get('Kecamatan', r.get('kecamatan', 'Cilacap'))).strip()
    lat_desa = float(r['Latitude'] if 'Latitude' in r else r['latitude'])
    lon_desa = float(r['Longitude'] if 'Longitude' in r else r['longitude'])
    elevasi = float(r.get('elevasi_m', 0))
    slope = float(r.get('slope_deg', 0))
    curah_hujan = float(r.get('rr', 0))
    skor_risiko = float(r.get('skor_risiko', 0))
    kelas_risiko = str(r.get('kelas', 'Rendah')).strip()
    
    # 🔍 STRATEGI SEARCH: Cari secara paksa posko mana yang jaraknya paling dekat dari desa ini
    posko_terdekat = None
    jarak_terminimum = float('inf')
    
    for sh in shelter_list:
        d = hitung_jarak(lat_desa, lon_desa, sh["lat"], sh["lon"])
        if d < jarak_terminimum:
            jarak_terminimum = d
            posko_terdekat = sh

    list_desa_rawan.append({
        "desa": nama_desa, 
        "kecamatan": kecamatan,
        "lat": lat_desa,
        "lon": lon_desa,
        "skor_risiko": skor_risiko,
        "kelas": kelas_risiko,
        "elevasi_m": elevasi,
        "slope_deg": slope,
        "rr": curah_hujan,
        "warga": 0
    })
    
    # Masukkan rute baru hasil pencarian jarak terdekat
    if posko_terdekat:
        list_jalur.append({
            "from": {"name": nama_desa, "kec": kecamatan, "lat": lat_desa, "lon": lon_desa},
            "to": {"name": posko_terdekat["nama"], "lat": posko_terdekat["lat"], "lon": posko_terdekat["lon"]},
            "dist_km": round(jarak_terminimum, 2), # Jarak riil terdekat hasil hitung rumus Haversine
            "skor_risiko": skor_risiko,
            "kelas": kelas_risiko,
            "elevasi_m": elevasi,
            "slope_deg": slope,
            "rr": curah_hujan
        })

# ── 3. BUNDLE DATA JAVASCRIPT ─────────────────────────────────────────────
web_data = {
    "feat_importance": [{"name": "Curah Hujan (Real-time BMKG)", "pct": 45.0}, {"name": "Ketinggian Lahan (DEM)", "pct": 30.0}, {"name": "Kemiringan Lereng (Slope)", "pct": 25.0}],
    "evakuasi": shelter_list,
    "jalur": list_jalur,
    "desa_rawan": list_desa_rawan
}

json_string = json.dumps(web_data, ensure_ascii=False, indent=2)

with open(PATH_INDEX_HTML, 'r', encoding='utf-8') as f:
    html_content = f.read()

start_marker = "const DATA = "
if start_marker in html_content:
    idx_start = html_content.find(start_marker)
    before_data = html_content[:idx_start + len(start_marker)]
    idx_end = html_content.find("</script>", idx_start)
    idx_close_bracket = html_content.rfind(";", idx_start, idx_end)
    after_data = html_content[idx_close_bracket:]
    
    with open(PATH_INDEX_HTML, 'w', encoding='utf-8') as f:
        f.write(before_data + json_string + after_data)
    print("✓ SELESAI! Semua rute desa sekarang otomatis ditarik ke balai desa terdekat.")
else:
    print("❌ Marker 'const DATA = ' tidak ditemukan!")

print("=" * 60)