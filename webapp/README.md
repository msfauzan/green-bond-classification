# Web App — Klasifikasi EBUS GSS

Antarmuka Streamlit untuk pipeline klasifikasi GSS (lihat `../classifier/`).

## Jalankan

```bash
pip install -r webapp/requirements.txt
streamlit run webapp/app.py
```

Buka `http://localhost:8501`. Model semantik (~120 MB) diunduh sekali saat
pertama kali dipakai, lalu di-cache.

## Tab

1. **Klasifikasi Prospektus** — unggah PDF prospektus (atau tempel teks), pilih
   kode emiten (opsional, mengaktifkan *title-lookup*) → kelas GSS, sektor
   eligible, skor keyakinan, dan bukti penjelasan.
2. **Statistik Pasar GSS** — scan *title-lookup* ke seluruh 1.437 EBUS di listing
   IDX: jumlah & nilai GSS per tipe, daftar instrumen terdeteksi.
3. **Evaluasi Model** — metrik P/R/F1 (rule-based vs ML) terhadap gold set, plus
   tabel hasil per dokumen.

Semua berjalan lokal & bebas — tanpa API berbayar.
