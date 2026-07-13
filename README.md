# Green Bond Classification — Bank Indonesia DSta-DSMF

Klasifikasi otomatis EBUS (Efek Bersifat Utang & Sukuk) korporasi Indonesia ke kategori
**GSS (Green / Social / Sustainability)** berdasarkan **POJK 18/2023**, ICMA, dan
kerangka sovereign DJPPR. Deliverable utamanya adalah **Analytical Note** — kode di repo
ini dipakai untuk mengumpulkan bukti dan membuat prototipe classifier.

## 4 Kategori

| Label | Keterangan |
|---|---|
| Green Bond | Obligasi Hijau — dana untuk proyek lingkungan |
| Sustainability Bond | Obligasi Keberlanjutan — lingkungan + sosial |
| Sustainability-Linked Bond | Obligasi Terkait Keberlanjutan — ada KPI/SPT + step-up coupon |
| Obligasi Biasa | Tidak termasuk ketiga kategori di atas |

## Instalasi (Laptop Baru)

### Prasyarat

- **Python 3.10+** — <https://www.python.org/downloads/> (centang *Add Python to PATH* saat install)
- **Git** — <https://git-scm.com/downloads>
- **Google Chrome** — wajib untuk scraper IDX (Selenium, situs di balik Cloudflare)

### 1. Clone repo

```bash
git clone https://github.com/msfauzan/green-bond-classification.git
cd green-bond-classification
```

### 2. Buat virtual environment (opsional tapi disarankan)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
```

### 3. Install dependensi per modul

Tidak ada `requirements.txt` di root — install sesuai modul yang mau dipakai:

```bash
# Web app / dashboard (Streamlit) — juga mencakup classifier
pip install -r code/webapp/requirements.txt

# Scraper listing obligasi IDX
pip install -r code/idx_bond_scraper/requirements.txt

# Scraper prospektus PDF (menambahkan PyMuPDF)
pip install -r code/idx_prospektus_scraper/requirements.txt
```

### 4. Data yang TIDAK ikut di repo (gitignored)

`*.pdf`, `*.xlsx`, `pdf_by_content/` (korpus prospektus hasil kurasi), dan `ML_Dataset/`
**tidak** ada di GitHub — pindahkan manual dari laptop lama (flashdisk/drive) ke lokasi
yang sama. Tanpa korpus PDF, classifier dan evaluasi tidak bisa jalan; scraper dan
dashboard statistik pasar tetap bisa.

> ⚠️ **Windows path limit 260 karakter** sering terlampaui oleh path korpus.
> Aktifkan long paths lalu **reboot**:
> ```powershell
> # jalankan PowerShell sebagai Administrator
> Set-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1
> git config --global core.longpaths true
> ```

## Cara Menjalankan

```bash
# Dashboard web (Streamlit)
streamlit run code/webapp/app.py

# Scrape listing obligasi/sukuk IDX (biarkan jendela Chrome terlihat — Cloudflare)
cd code/idx_bond_scraper && python scrape_idx_bonds.py

# Scrape prospektus PDF (default: 15 kode emiten GSS)
cd code/idx_prospektus_scraper && python scrape_prospektus.py

# Demo classifier rule-based terhadap prospektus terverifikasi
python code/idx_prospektus_scraper/demo_classify.py

# Inventaris korpus terkurasi
python code/idx_prospektus_scraper/list_all.py
```

## Struktur Direktori

```
├── code/
│   ├── classifier/            # Taksonomi (9 green + 6 social) & mesin klasifikasi
│   ├── evaluation/            # Skrip evaluasi & pembuatan dokumen AN
│   ├── idx_bond_scraper/      # Scraper listing obligasi/sukuk IDX
│   ├── idx_prospektus_scraper/# Scraper prospektus PDF + alat kurasi korpus
│   └── webapp/                # Dashboard Streamlit
├── data/                      # CSV kecil: lookup SBN (ground truth), hasil scan, dsb.
├── Deliverable AN GSS/        # Analytical Note: dokumen, data, grafik
├── materi/                    # Paper referensi, materi FGD, PPT
├── screenshot hasil kerja/    # Screenshot grafik & dashboard
└── pdf_by_content/            # Korpus prospektus PDF (gitignored — pindah manual)
```

## Dokumen Kunci

Baca dua dokumen ini sebelum mengubah metodologi — semuanya mengacu ke sana:

- [`Deliverable AN GSS/Kerangka_AN_Klasifikasi_GSS_ML.md`](Deliverable%20AN%20GSS/Kerangka_AN_Klasifikasi_GSS_ML.md) — kerangka AN (scope, metodologi, tahapan)
- [`Deliverable AN GSS/Brainstorming_AN_GSS.md`](Deliverable%20AN%20GSS/Brainstorming_AN_GSS.md) — reasoning & open questions
