# Paket Deliverable — Analytical Note Klasifikasi GSS EBUS Korporasi

**Bank Indonesia · DSta / DSMF** · Paket siap paparan pemangku kepentingan.

Klasifikasi efek bersifat utang & sukuk (EBUS) korporasi Indonesia ke dalam kategori
**Green / Social / Sustainability (GSS)** per **POJK 18/2023**, ICMA, dan kerangka DJPPR.

---

## Ringkasan eksekutif

| Indikator | Nilai |
|---|---|
| Universe EBUS korporasi | 1,436 instrumen |
| GSS berlabel (sensus judul IDX) | 82 instrumen (5.71% universe) |
| Total outstanding GSS | Rp 62.21 T |
| Akurasi classifier final (ML semantik) | P 1.000 · R 1.000 · F1 1.000 |
| Baseline rule-based | P 0.333 · R 0.852 · F1 0.479 |
| Gold set evaluasi | 93 dokumen terverifikasi |

**Temuan kunci:** mayoritas EBUS korporasi (94.29%) belum terklasifikasi GSS —
inilah celah yang diisi oleh classifier taksonomi-grounded yang dapat diaudit
(explainable by design, model lokal gratis, tanpa LLM berbayar).

---

## Struktur paket

```
Deliverable AN GSS/
├── 01_Dokumen/   Dokumen AN lengkap (.docx untuk dibaca/edit, .md sumber)
├── 02_Data/      CSV bukti — semua angka di AN & grafik berasal dari sini
├── 03_Grafik/    8 grafik PNG siap tempel ke paparan / dokumen
└── README.md     berkas ini
```

### 01_Dokumen
- `Analytical_Note_Klasifikasi_GSS_EBUS_Korporasi.docx` — naskah AN lengkap (§0–§12).

### 02_Data
| Berkas | Isi |
|---|---|
| `01_sensus_pasar_gss.csv` | Sensus 82 GSS berlabel per kelas + outstanding + share universe |
| `02_dekomposisi_sektoral.csv` | Dekomposisi use-of-proceeds 27 gold GSS (per bond) |
| `03_hasil_evaluasi_klasifikasi.csv` | Confusion matrix rule vs ML pada gold set |
| `04_daftar_82_gss_berlabel.csv` | Daftar 82 instrumen GSS hasil sensus judul IDX |
| `05_kandidat_gss_tidak_berlabel.csv` | Kandidat GSS tidak berlabel dari pemindaian semesta |
| `06_statistik_deskriptif_gss.csv` | Penerbitan per tahun, profil jatuh tempo, konsentrasi (HHI) & rating |

### 03_Grafik
| Berkas | Menjelaskan |
|---|---|
| `01_sensus_pasar_per_kelas.png` | Jumlah & outstanding GSS per kelas (Green/Social/SL) |
| `02_gap_labeled_vs_universe.png` | Celah 82 berlabel vs 1,436 universe EBUS |
| `03_evaluasi_rule_vs_ml.png` | Precision/Recall/F1 rule-based vs ML |
| `04_dekomposisi_sektoral.png` | Status dekomposisi + kategori UoP yang didanai |
| `05_cakupan_pemindaian_semesta.png` | Cakupan scan universe: berlabel / kandidat / dipindai / belum |
| `06_tren_penerbitan_gss.png` | Tren penerbitan GSS per tahun (2022–2026), jumlah + outstanding |
| `07_profil_jatuh_tempo.png` | Profil jatuh tempo outstanding GSS (maturity ladder) |
| `08_konsentrasi_dan_rating.png` | Konsentrasi penerbit (HHI, top-5) & sebaran rating |

---

## Metodologi (ringkas)

Pipeline berlapis & dapat diaudit:
1. **Aturan Level-0** — struktur instrumen (Sustainability-Linked: KPI/SPT/step-up; Sukuk Wakaf).
2. **Title-gate** — pencocokan judul dari listing IDX (ambang adaptif).
3. **Framing-gate** — frasa POJK 18/2023 ("berwawasan lingkungan", dll.), bergerbang-negasi.
4. **Kemiripan semantik** — sentence-transformer lokal `paraphrase-multilingual-MiniLM-L12-v2`
   (**gratis, offline, tanpa API berbayar**).
5. **Fallback judul→kelas** — bila UoP terlalu tipis, tandai `needs_review` /
   `sector_unspecified` (kelas terkonfirmasi dari nama, sektor tak terverifikasi — dilaporkan jujur).

Setiap keputusan menyebut kriteria/kata kunci pemicunya → **explainable by design**.

---

## Catatan kejujuran metodologis

- Dekomposisi sektoral menggunakan pendekatan **hibrida leksikal+semantik**: sektor dihitung
  hanya bila skor kosinus ≥ ambang gate **DAN** ≥1 kata kunci taksonomi kategori tersebut
  hadir di teks UoP. Ini memisahkan gate biner (maks recall) dari dekomposisi (maks presisi).
- Sebagian bond bank-intermediary mencantumkan seluruh 15 kategori dalam kerangka framework
  → kata kunci semua kategori hadir di UoP → kategori "didanai" merefleksikan cakupan
  kerangka, bukan realisasi alokasi aktual. Ini adalah keterbatasan data sumber, bukan
  kesalahan classifier.
- Kolom `verification_status` menandai konsistensi antara klaim (nama instrumen) dan
  bukti UoP: Terverifikasi / Sebagian / Tidak konsisten / Tidak tersubstansiasi.
- Gold set masih timpang antar-kelas (Green/SL sedikit) → akurasi sempurna pada
  gold set bukan jaminan generalisasi penuh.

*Dihasilkan otomatis oleh `evaluation/make_deliverable.py` — regenerasikan bila data berubah.*
