# Green Bond Classification — Bank Indonesia DSta-DSMF

Klasifikasi otomatis obligasi/sukuk berdasarkan **POJK 18/2023** (Efek Bersifat Utang dan/atau Sukuk Berwawasan Lingkungan).

## 4 Kategori

| Label | Keterangan |
|---|---|
| Green Bond | Obligasi Hijau — dana untuk proyek lingkungan |
| Sustainability Bond | Obligasi Keberlanjutan — lingkungan + sosial |
| Sustainability-Linked Bond | Obligasi Terkait Keberlanjutan — ada KPI/IKU + step-up coupon |
| Obligasi Biasa | Tidak termasuk ketiga kategori di atas |

## Struktur Direktori

```
├── pdf_by_content/          # PDF prospektus (diorganisir per jenis konten)
├── ML_Dataset/              # Dataset berlabel (CSV) untuk training ML
├── klasifikasi prospektus/  # Sampel PDF yang sudah diklasifikasi (ground truth)
├── prospektus_by_category/  # PDF diorganisir per kategori obligasi
├── FGD AN Green Debt Securities/  # Materi & risalah FGD
├── Paper Referensi ML/      # Paper referensi untuk pendekatan ML
├── data/                    # Data analisis & laporan
├── List Perusahaan/         # Daftar perusahaan emiten
└── Tabel Referensi AN Green Bond.xlsx  # Tabel referensi analisis
```
