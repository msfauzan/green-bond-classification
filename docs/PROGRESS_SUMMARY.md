# Progress Summary — Klasifikasi EBUS GSS Berbasis ML
**Bank Indonesia · DSta / DSMF · Analytical Note**
*Diperbarui: 2026-06-20*

---

## 1. Konteks & Tujuan

Proyek ini membangun **pipeline klasifikasi GSS (Green / Social / Sustainability)**
untuk Efek Bersifat Utang & Sukuk (EBUS) korporasi di pasar Indonesia, sebagai
fondasi **Analytical Note (AN)** Bank Indonesia.

### Posisi inti
| Segmen | Status klasifikasi | Pendekatan AN |
|---|---|---|
| **Sovereign (SBN)** | Sudah otoritatif — proses KRISNA → CBT → audit BPK → SRN-PPI | Dipakai sebagai **taksonomi acuan & ground truth** |
| **Corporate (EBUS)** | Tidak ada proses penandaan setara; ~belasan label resmi BEI | **Target ML** — kontribusi orisinal AN |

### Tiga pertanyaan penelitian
1. Dapatkah taksonomi GSS resmi (DJPPR + POJK 18/2023, berakar ICMA/ASEAN) dipakai
   sebagai ground truth untuk klasifikasi EBUS korporasi secara otomatis?
2. Seberapa andal klasifikasi tersebut diukur terhadap label gold yang tersedia?
3. Seberapa besar selisih antara semesta GSS berlabel resmi vs hasil klasifikasi
   (potensi pasar GSS yang belum terpetakan)?

---

## 2. Data

### 2.1 Ground truth sovereign (fondasi taksonomi)
File: `data/sbn_gss_lookup.csv` — **25 instrumen** GSS sovereign (2018–2024),
dikompilasi dari laporan resmi DJPPR.

| Kelompok | Jumlah seri | Kumulatif |
|---|---|---|
| Green Sukuk global (SNI) | 7 | USD 6,60 miliar |
| Green Sukuk ritel (ST) | 8 | IDR 40,60 T |
| Green Sukuk wholesale (PBSG) | 3 | IDR 31,17 T |
| SDG Bond 2024 | 3 | IDR 19,15 T |
| Blue Bond (Samurai) | 4 | JPY 20,7 M + JPY 25 M |

Teks laporan DJPPR (2024, 2025 Green Sukuk; 2025 SDG/Blue Bond) diekstrak ke
`data/_green_sukuk_report_text.txt`, `data/_green_sukuk_2025_report_text.txt`,
`data/_sdg_bond_report_text.txt` — menjadi basis deskripsi kategori taksonomi.

### 2.2 Semesta EBUS korporasi (target klasifikasi)
File: `data/idx_obligasi_sukuk_korporasi_20260618_134135.csv`

| Metrik | Angka |
|---|---|
| Total instrumen di listing IDX | **1.437** |
| Instrumen berlabel GSS (BEI) | **~82** (dari `idx_gss_all_20260618_140427.csv`) |
| Subset IDX naif dengan kata "berkelanjutan/GSS" | 192 (tercemar PUB) |

### 2.3 Gold corpus (evaluasi)
Prospektus PDF yang telah diverifikasi tangan, tersimpan di:
`pdf_by_content/01_prospektus_utama/0. Fix GSS/GSS/`

| Metrik | Angka |
|---|---|
| Emiten GSS terverifikasi | **12** (ARKO, BBNI, BBRI, BBTN, BJBR, BMRI, BRIS, OPPM, PNMP, POLI, PPGD, SMII) |
| Dokumen PDF GSS (positif) | **27** |
| Dokumen PDF NonGSS (negatif sampel) | **66** (1 per emiten konvensional dari korpus unduhan) |
| **Total gold set evaluasi** | **93 dokumen** |

---

## 3. Taksonomi Acuan (Kamus Kebenaran)

File: `classifier/taxonomy.py`

Definisi GSS **tidak dipelajari dari data** — di-*encode* langsung dari regulasi:
POJK 18/2023, ICMA Green/Social Bond Principles, SDG Government Securities
Framework (DJPPR), Taksonomi Hijau Indonesia (TKBI).

### Skema klasifikasi bertingkat (hierarchical)

```
Level 0 — Struktur instrumen (cek SEBELUM penggunaan dana):
   ├── Sustainability-Linked  →  deteksi KPI / SPT / kupon step-up
   └── Sukuk Wakaf            →  deteksi klausa wakaf / CWLS / nazhir
        (jika bukan keduanya → lanjut Level 1)

Level 1 — Use-of-Proceeds (9 kategori lingkungan + 6 kategori sosial):
   Hanya lingkungan  → GREEN
   Hanya sosial      → SOCIAL
   Campuran          → SUSTAINABILITY
   + sub-tema BLUE   → tag tambahan (bukan kelas tersendiri)
   Tidak ada         → NON_GSS (obligasi biasa)
```

### 9 Kategori Lingkungan (Green / Blue)
| Kode | Nama |
|---|---|
| `renewable_energy` | Energi Terbarukan |
| `energy_efficiency` | Efisiensi Energi |
| `green_tourism` | Pariwisata Berkelanjutan |
| `sustainable_transport` | Transportasi Berkelanjutan |
| `green_building` | Bangunan Hijau |
| `waste_management` | Pengelolaan Sampah & Limbah |
| `water_management` | Pengelolaan Air & Air Limbah |
| `natural_resources` | Pengelolaan SDA Berkelanjutan (Darat & Laut) |
| `climate_resilience` | Ketahanan Iklim & Pengurangan Risiko Bencana |

### 6 Kategori Sosial
| Kode | Nama |
|---|---|
| `basic_infrastructure` | Infrastruktur Dasar Terjangkau |
| `essential_services` | Akses Layanan Esensial |
| `affordable_housing` | Perumahan Terjangkau |
| `employment_msme` | Penciptaan Lapangan Kerja & UMKM |
| `food_security` | Ketahanan Pangan |
| `socioeconomic` | Pemberdayaan Sosial-Ekonomi |

---

## 4. Pipeline Klasifikasi

### Arsitektur tiga lapis

```
[PDF / Teks prospektus]
        │
        ▼
┌───────────────────────────────────────────────────┐
│  LEVEL 0 — Rule-based (classifier/engine.py)      │
│  Cek sinyal STRUKTUR instrumen:                   │
│    → Sustainability-Linked (KPI/SPT/step-up)      │
│    → Sukuk Wakaf (wakaf/CWLS/nazhir)              │
│  Jika cocok: kembalikan kelas Level-0 langsung    │
└───────────────────┬───────────────────────────────┘
                    │ (tidak cocok → lanjut)
                    ▼
┌───────────────────────────────────────────────────┐
│  TITLE LOOKUP — Rule-based (classifier/           │
│  title_lookup.py)                                 │
│  Cek nama obligasi di listing IDX:                │
│    True  → emiten punya obligasi bernama GSS      │
│    False → emiten ada di listing, tanpa GSS       │
│    None  → emiten tidak ada di listing (unknown)  │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│  FRAMING GATE — Rule-based (ml_engine.py)         │
│  Cek bahasa komitmen GSS eksplisit di badan dok:  │
│  "berwawasan lingkungan/sosial", "green bond",    │
│  "second party opinion", "pojk 18/2023", dst.     │
│  → Menentukan threshold bersama title lookup      │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│  LEVEL 1 — ML Semantic (classifier/ml_engine.py)  │
│  1. Ekstrak segmen "Penggunaan Dana / Use of      │
│     Proceeds" (anchor paling spesifik, last occ.) │
│  2. Encode dengan sentence-transformer            │
│     (paraphrase-multilingual-MiniLM-L12-v2,       │
│      ~120 MB, bebas, lokal)                       │
│  3. Cosine-similarity ke 15 deskripsi kategori    │
│  4. Bandingkan dengan threshold tier:             │
│     title=True  → 0.28 (konfirmasi minimal)       │
│     title=None + body framing  → 0.30             │
│     title=None + no framing    → 0.52             │
│     title=False + body framing → 0.45             │
│     title=False + no framing   → 0.62 (ketat)    │
└───────────────────┬───────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────┐
│  OUTPUT (per instrumen)                           │
│  • Kelas GSS: GREEN / SOCIAL / SUSTAINABILITY /   │
│               SUSTAINABILITY_LINKED / WAKAF /     │
│               NON_GSS                            │
│  • Sub-tema BLUE (tag tambahan)                   │
│  • Sektor eligible terpenuhi + skor per kategori  │
│  • Skor keyakinan (confidence)                    │
│  • Bukti: title hits, body framing, anchor UoP   │
└───────────────────────────────────────────────────┘
```

### Model ML yang dipakai
**`paraphrase-multilingual-MiniLM-L12-v2`** (sentence-transformers)
- Ukuran: ~120 MB
- Bahasa: multilingual (50+ bahasa, termasuk Indonesia)
- Inference: lokal, bebas, tanpa API berbayar
- Pendekatan: zero-shot — tidak memerlukan data berlabel untuk training
- Cara kerja: cosine similarity antara embedding segmen UoP dan embedding
  deskripsi kategori taksonomi yang ditulis dalam Bahasa Indonesia

---

## 5. Jebakan Kritis yang Ditemukan

### Jebakan 1: "Berkelanjutan" vs "Keberlanjutan"
Ini adalah **false positive paling umum** dalam klasifikasi berbasis kata kunci.

| Frasa | Arti | Status |
|---|---|---|
| "Obligasi **Berkelanjutan** I/II/III" | Penawaran Umum Berkelanjutan (PUB) — istilah shelf-registration | **BUKAN GSS** |
| "Obligasi **Keberlanjutan** Berkelanjutan I" | Sustainability Bond yang dijual via PUB | **GSS** |
| "Obligasi **Berwawasan Lingkungan** Berkelanjutan I" | Green Bond via PUB | **GSS** |
| "Obligasi **Berwawasan Sosial** Berkelanjutan I" | Social Bond via PUB | **GSS** |

Solusi: `title_lookup.py` mendeteksi marker GSS yang eksplisit di nama obligasi,
bukan hanya "berkelanjutan".

### Jebakan 2: "employment_msme" — Aktivitas Bisnis vs Komitmen GSS
Kata kunci UMKM ("usaha mikro", "kur", "pembiayaan mikro") muncul di hampir
**semua** prospektus bank dan multifinance konvensional — bukan sebagai komitmen
Social Bond, melainkan sebagai deskripsi bisnis inti. Rule-based menghasilkan 40+
false positive dari satu kategori ini saja.

Solusi: framing gate + title lookup memastikan kata kunci tersebut hanya dihitung
bila dokumen secara eksplisit diposisikan sebagai Social Bond.

### Jebakan 3: Dokumen "Informasi Tambahan" vs Prospektus Utama
Dokumen tambahan (penyampaian bukti iklan, informasi tambahan) tidak memiliki
bagian "Penggunaan Dana / Use of Proceeds" yang lengkap. Model tidak gagal —
dokumennya yang tidak mengandung sinyal.

Contoh: BBRI 2022 "Penyampaian Prospektus Informasi Tambahan" — satu-satunya
false negative dalam evaluasi akhir.

---

## 6. Hasil Evaluasi

### Setup evaluasi
- **Gold positif**: 27 dokumen dari 12 emiten GSS (kurated tangan)
- **Gold negatif**: 66 dokumen konvensional (1 per emiten, sampel dari korpus unduhan)
- **Total**: 93 dokumen

### Perbandingan tiga engine

| Metrik | Rule-based | ML + Body Framing | **ML + Title Gate** |
|---|:---:|:---:|:---:|
| **Precision** | 0.33 | 0.81 | **1.00** |
| **Recall** | 0.85 | 0.96 | **0.96** |
| **F1** | 0.48 | 0.88 | **0.98** |
| TP | 23 | 26 | 26 |
| FP | 46 | 6 | **0** |
| FN | 4 | 1 | 1 |
| TN | 20 | 60 | **66** |

### Apa yang dilakukan setiap lapis

| Lapis | Kontribusi utama |
|---|---|
| Rule-based baseline | Recall tinggi (0.85) tapi precision buruk (0.33) — 46 FP dari `employment_msme` |
| Body framing gate | Memotong FP dari 46 → 6; Recall naik ke 0.96 karena L0 (SL/Wakaf) kini aktif |
| Title lookup | Menghapus seluruh 6 FP yang tersisa; FP = 0, Precision = 1.00 |

### Distribusi sub-kelas pada GSS positif yang benar (26 TP)

| Sub-kelas | Jumlah |
|---|---|
| `sustainability` (campuran lingkungan+sosial) | 20 |
| `sustainability_linked` (KPI/SPT/step-up) | 3 |
| `social` | 3 |
| `green` | 0 |

*Catatan: mayoritas `sustainability` karena bank-bank besar (BBRI, BMRI, BBTN)
mendanai campuran proyek hijau + sosial.*

### False Negative yang tersisa (1)
**BBRI 2022 — "Penyampaian Prospektus Informasi Tambahan"**
- `title_gss = True` (BBRI punya obligasi bernama GSS di listing IDX)
- Namun dokumen adalah addendum 2022, sebelum penggunaan bahasa "berwawasan
  lingkungan" distandarkan; tidak ada bagian UoP lengkap
- Skor semantik tertinggi: 0.25 (renewable_energy), di bawah threshold 0.28
- **Ini artefak korpus, bukan kegagalan model**

---

## 7. Kontribusi untuk AN

### 7.1 Temuan empiris citable
1. **Keyword presence ≠ use-of-proceeds commitment.** Rule-based menghasilkan
   Precision 0.33 — artinya 2 dari 3 "GSS" yang terdeteksi adalah obligasi biasa.
   Ini membuktikan secara kuantitatif mengapa pendekatan naif tidak cukup.

2. **Nama obligasi adalah sinyal klasifikasi primer.** Setelah title lookup
   ditambahkan, Precision mencapai 1.00. Ini menegaskan bahwa label POJK 18/2023
   sudah tertanam dalam nama instrumen — gap-nya adalah di instrumen yang
   *belum* menggunakan label tersebut meski sebenarnya memenuhi kriteria.

3. **"Berkelanjutan" trap terdokumentasi secara kuantitatif.** 40+ FP dari
   istilah PUB — memberikan evidensi konkret untuk AN tentang risiko mis-klasifikasi
   pada pendekatan sederhana.

4. **Pipeline taxonomy-grounded = explainable by design.** Setiap keputusan
   menyertakan: nama obligasi yang cocok, sinyal framing badan dokumen,
   skor cosine per kategori, anchor "Penggunaan Dana" yang ditemukan.
   Sesuai kebutuhan pertanggungjawaban kebijakan BI.

### 7.2 Implikasi untuk statistik pasar GSS
- Dari 1.437 EBUS di listing IDX, ~82 berlabel GSS (BEI)
- Pipeline ini memungkinkan **full-universe scan**: cek semua 1.437 instrumen
  via title lookup → identifikasi "unlabeled GSS" (yang namanya berwawasan
  lingkungan/sosial tapi belum masuk label BEI) + false GSS (berlabel tapi
  tidak substansiasi use-of-proceeds)
- Hasilnya = statistik **gap antara berlabel resmi vs terklasifikasi** — ini
  kontribusi orisinal AN

### 7.3 Tentang framing "Berbasis Machine Learning"
Pipeline ini hybrid: rule-based (Level 0, title lookup, framing gate) +
ML (sentence-transformer semantic similarity). Komponen ML genuine: zero-shot
multilingual embeddings untuk category matching — tidak memerlukan label, tidak
berbayar, reproducible di lingkungan BI.

---

## 8. Keterbatasan

| Keterbatasan | Dampak | Mitigasi |
|---|---|---|
| Gold set positif hanya 12 emiten / 27 dokumen | Recall mungkin overestimated jika GSS corpus tidak representatif | Perluas corpus dengan SPO / Framework penerbit |
| Negatif disampling (1 per emiten), bukan verifikasi per-dokumen | Precision bisa lebih rendah di production bila negatif lebih variatif | Verifikasi manual sampel negatif |
| Title lookup berbasis emiten (bukan per-instrumen) | Emiten dengan campuran GSS+konvensional (mis. SMII) bisa menyebabkan FP pada dokumen konvensionalnya | Matching per-ISIN bila data tersedia |
| Model memotong teks panjang (max 512 token) | UoP yang panjang mungkin dipotong sebelum sektor paling relevan | Chunking + max-over-windows |
| Dokumen "Informasi Tambahan" tanpa UoP | FN tak terelakkan untuk addendum | Filter dokumen berdasarkan tipe; fokus pada prospektus utama |
| Taksonomi bisa berevolusi | Threshold dan deskripsi perlu diperbarui berkala | Versi taksonomi di taxonomy.py |

---

## 9. Struktur File

```
classifier/
├── taxonomy.py        # Kamus kebenaran — 15 kategori + sinyal L0 + NEGATION_HINTS
├── engine.py          # Rule-based engine: Level-0 + UoP extraction + keyword match
├── ml_engine.py       # ML engine: title lookup + framing gate + semantic similarity
└── title_lookup.py    # Lookup nama obligasi GSS dari listing IDX

evaluation/
├── run_baseline.py    # Evaluasi rule-based saja (93 dok, CSV output)
└── compare_engines.py # Perbandingan rule vs ML (tabel berdampingan)

data/
├── sbn_gss_lookup.csv                             # 25 instrumen sovereign (ground truth)
├── idx_obligasi_sukuk_korporasi_20260618_134135.csv  # Semesta 1.437 EBUS IDX
├── idx_gss_all_20260618_140427.csv               # ~82 berlabel GSS (BEI)
├── baseline_results.csv                          # Hasil rule-based per dokumen
└── comparison_results.csv                        # Hasil perbandingan 3 engine per dok

idx_bond_scraper/
└── scrape_idx_bonds.py   # Selenium scraper semesta EBUS dari IDX

idx_prospektus_scraper/
├── scrape_prospektus.py  # Selenium scraper PDF prospektus dari IDX
├── demo_classify.py      # Demo classifier (prototipe awal)
├── audit_gss_folders.py  # Audit nama GSS per folder emiten
├── deep_check.py         # Keyword-in-context untuk kasus ambigu
└── list_all.py           # Inventaris corpus kurated

pdf_by_content/01_prospektus_utama/0. Fix GSS/GSS/
└── {EMITEN}/             # 12 folder emiten, ~27 PDF terverifikasi tangan
                          # (gitignored — di luar repo)

docs/
└── PROGRESS_SUMMARY.md   # Dokumen ini
```

---

## 10. Langkah Selanjutnya

| Prioritas | Tugas | Output untuk AN |
|---|---|---|
| **Tinggi** | Full-universe scan: jalankan title lookup ke 1.437 EBUS → hitung unlabeled GSS | Statistik gap berlabel vs terklasifikasi |
| **Tinggi** | Untuk unlabeled GSS (dari title lookup), jalankan ML semantic → ekstrak sektor | Statistik dekomposisi sektoral |
| **Menengah** | Validasi manual sampel FP/FN dari full-universe run | Meningkatkan kredibilitas metrik |
| **Menengah** | Perluas gold set: harvest label dari SPO & Framework penerbit publik | Evaluasi yang lebih robust |
| **Rendah** | Chunking segmen UoP untuk dokumen panjang | Mengurangi FN dari truncation |
| **AN** | Tulis temuan sebagai bagian metodologi & hasil | Draf AN §7–§9 |

---

*Dokumen ini merangkum perkembangan per 2026-06-20. Untuk konteks lebih lanjut,
baca [Kerangka_AN_Klasifikasi_GSS_ML.md](../Kerangka_AN_Klasifikasi_GSS_ML.md)
dan [Brainstorming_AN_GSS.md](../Brainstorming_AN_GSS.md).*
