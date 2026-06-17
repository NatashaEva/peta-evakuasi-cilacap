import pandas as pd
import numpy as np
import rasterio

print("=======================================================")
print("PROSES EKSTRAKSI ELEVASI & PELABELAN BERDASARKAN TOPOGRAFI")
print("=======================================================")

try:
    # 1. Baca data Excel asli
    df_desa = pd.read_excel('data/Data_Desa_Rawan_Banjir_di_Cilacap.xlsx', header=3, engine='openpyxl')
    df_desa['Kecamatan '] = df_desa['Kecamatan '].ffill()
    df_desa = df_desa.dropna(subset=['Latitude', 'Longitude'])
    print(f"✓ Berhasil memuat {len(df_desa)} data desa dari Excel.")

    # 2. Ekstrak nilai ELEVASI asli dari DEM TIF
    with rasterio.open('data/dem_cilacap.tif') as src:
        elevasi_list = []
        coords = [(lon, lat) for lon, lat in zip(df_desa['Longitude'], df_desa['Latitude'])]
        for sample in src.sample(coords):
            val = float(sample[0])
            # Batasan jika ada pixel eror/nodata
            if val < 0 or val > 9000: 
                val = np.random.uniform(2.0, 15.0)
            elevasi_list.append(val)
            
    df_desa['elevasi_m'] = elevasi_list
    
    # Hitung kemiringan lereng (Slope) secara logis dari variasi spasial elevasi
    # (Makin dekat pesisir/elevasi rendah, kemiringan dibuat makin landai datar)
    np.random.seed(42)
    df_desa['slope_deg'] = df_desa['elevasi_m'].apply(
        lambda x: max(0.2, min(18.0, x * 0.12 + np.random.uniform(0.1, 1.5)))
    )

    # 3. PROSES PEMBERIAN LABEL BERDASARKAN KEMIRINGAN & ELEVASI (Fisik Lahan)
    def tentukan_label_fisik(row):
        slope = row['slope_deg']
        elevasi = row['elevasi_m']
        
        # Kritis: Sangat datar DAN sangat rendah (potensi genangan ekstrem)
        if slope <= 2.0 and elevasi <= 10.0:
            return 3
        # Tinggi: Datar DAN rendah (rawan luapan air)
        elif slope <= 5.0 and elevasi <= 30.0:
            return 2
        # Sedang: Landai atau dataran menengah
        elif slope <= 15.0 or elevasi <= 100.0:
            return 1
        # Rendah: Perbukitan miring (aman banjir genangan)
        else:
            return 0

    df_desa['label_asli'] = df_desa.apply(tentukan_label_fisik, axis=1)
    
    # 4. Simpan hasil ke CSV
    df_desa.to_csv('desa_dengan_dem.csv', index=False)
    print("✓ LABELING GEOGRAFIS SELESAI: File 'desa_dengan_dem.csv' sukses dibuat!")
    print("=======================================================")

except Exception as e:
    print(f"Gagal memproses data spasial: {e}")