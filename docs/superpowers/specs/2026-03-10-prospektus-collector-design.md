# Design Doc: Prospektus Collector — Efek Bersifat Utang (POJK 18/2023)

**Tanggal:** 2026-03-10
**Proyek:** Green Bond Classification — Bank Indonesia DSta-DSMF
**Status:** Approved

---

## Ringkasan

Aplikasi script manual berbasis Python yang memanfaatkan Firecrawl API untuk mengumpulkan seluruh PDF prospektus penawaran efek bersifat utang (obligasi, sukuk, EBA) di Indonesia yang diterbitkan setelah berlakunya POJK 18/2023. PDF dikumpulkan dari IDX, OJK, KSEI, dan website emiten (auto-discovered dari IDX), lalu disimpan ke Cloudflare R2.

---

## Arsitektur

### Struktur Folder

```
prospektus_collector/
├── orchestrator.py          # Entry point, koordinasi semua crawler
├── config.py                # API keys, filter tanggal, R2 config
├── tracker.py               # SQLite lokal untuk tracking & deduplication
├── uploaders/
│   └── r2_uploader.py       # Upload PDF ke Cloudflare R2
└── crawlers/
    ├── base_crawler.py      # Base class: Firecrawl client, PDF downloader
    ├── idx_crawler.py       # IDX: pengumuman + discover emiten list
    ├── ojk_crawler.py       # OJK: publikasi efek bersifat utang
    ├── ksei_crawler.py      # KSEI: keterbukaan informasi
    └── emiten_crawler.py    # Emiten: crawl IR/investor relations page
```

### Komponen Utama

| Komponen | Tanggung Jawab |
|---|---|
| `orchestrator.py` | Entry point; jalankan semua crawler secara async; cetak ringkasan |
| `config.py` | Semua konfigurasi (API keys, tanggal filter, R2 bucket, throttle) |
| `tracker.py` | SQLite DB lokal untuk catat PDF yang sudah diunduh (hindari duplikat) |
| `base_crawler.py` | Wrapper Firecrawl client; async PDF downloader; retry logic |
| `idx_crawler.py` | Scrape IDX pengumuman + ekstrak daftar emiten dan URL websitenya |
| `ojk_crawler.py` | Crawl seksi publikasi OJK untuk PDF prospektus |
| `ksei_crawler.py` | Scrape KSEI keterbukaan informasi untuk PDF prospektus |
| `emiten_crawler.py` | Terima list URL emiten; temukan halaman IR; ekstrak PDF |
| `r2_uploader.py` | Upload PDF ke Cloudflare R2 dengan folder structure terstruktur |

---

## Alur Data

```
orchestrator.py
    │
    ├─ idx_crawler.discover_emiten()
    │       └─ Firecrawl /map → profil emiten IDX
    │               └─ return: [(kode, url_website), ...]
    │
    ├─ asyncio.gather(
    │       idx_crawler.run(),
    │       ojk_crawler.run(),
    │       ksei_crawler.run(),
    │       emiten_crawler.run(emiten_list)
    │  )
    │
    └─ Per PDF link yang ditemukan:
            1. Cek tracker.db — skip jika sudah ada
            2. Download PDF via aiohttp (async)
            3. Rename: YYYYMMDD_KODE_filename.pdf
            4. Upload ke R2: prospektus/{sumber}/{kode_emiten}/
            5. Catat di tracker.db: url, filename, sumber, tanggal_crawl, r2_key, status
```

---

## Penggunaan Firecrawl per Sumber

### IDX (`idx.co.id`)
- **Discover emiten:** Firecrawl `/map` pada halaman daftar perusahaan tercatat → dapat URL profil dan website tiap emiten
- **Pengumuman:** Firecrawl `/scrape` per halaman pengumuman → filter keyword `prospektus`, `penawaran umum`, `efek bersifat utang`
- **Filter tanggal:** hanya dokumen ≥ tanggal berlaku POJK 18/2023

### OJK (`ojk.go.id`)
- Firecrawl `/crawl` pada seksi publikasi efek → enumerate semua halaman
- Filter PDF: nama file atau anchor text mengandung `prospektus` + `obligasi`/`sukuk`/`EBA`

### KSEI (`ksei.co.id`)
- Firecrawl `/scrape` dengan pagination → ekstrak PDF prospektus dari seksi keterbukaan informasi

### Emiten (website masing-masing)
- Input: list `(kode_emiten, url_website)` dari IDX discover
- Firecrawl `/map` per emiten → temukan halaman `investor-relations`, `hubungan-investor`, `publikasi`, `prospektus`
- Firecrawl `/scrape` halaman tersebut → ekstrak link PDF prospektus

---

## Struktur R2 Bucket

```
prospektus/
├── idx/
│   └── {KODE_EMITEN}/
│       └── YYYYMMDD_KODE_filename.pdf
├── ojk/
│   └── {KODE_EMITEN}/
│       └── YYYYMMDD_KODE_filename.pdf
├── ksei/
│   └── {KODE_EMITEN}/
│       └── YYYYMMDD_KODE_filename.pdf
└── emiten/
    └── {KODE_EMITEN}/
        └── YYYYMMDD_KODE_filename.pdf
```

---

## Konfigurasi (`config.py`)

```python
FIRECRAWL_API_KEY = ""          # Firecrawl API key
POJK_18_DATE = "2023-01-01"     # Filter minimum tanggal dokumen
CLOUDFLARE_R2_BUCKET = ""       # Nama R2 bucket
CLOUDFLARE_R2_ENDPOINT = ""     # R2 endpoint URL
CLOUDFLARE_ACCOUNT_ID = ""      # Cloudflare account ID
CLOUDFLARE_ACCESS_KEY_ID = ""   # R2 access key
CLOUDFLARE_SECRET_ACCESS_KEY = "" # R2 secret key
MAX_CONCURRENT_EMITEN = 10      # Throttle agar tidak kena rate limit
FIRECRAWL_TIMEOUT = 30          # Timeout per request (detik)
PDF_KEYWORDS = [                 # Keyword filter untuk PDF prospektus
    "prospektus", "penawaran umum", "efek bersifat utang",
    "obligasi", "sukuk", "eba", "medium term notes", "mtn"
]
```

---

## Error Handling

| Skenario | Penanganan |
|---|---|
| Firecrawl timeout/error | Retry 3x dengan exponential backoff, lalu skip & log |
| PDF download gagal | Catat di tracker dengan status `failed`, bisa di-retry dengan flag `--retry-failed` |
| Emiten tidak ada halaman IR | Skip & log, tidak crash keseluruhan |
| R2 upload gagal | PDF tersimpan di temp lokal, retry upload saat run berikutnya |
| Duplikat PDF | Skip berdasarkan URL atau hash file, catat di tracker |

---

## Output Script

```
=== PROSPEKTUS COLLECTOR ===
Filter tanggal : >= 2023-01-01 (POJK 18/2023)

[IDX]    Ditemukan 145 PDF | Diupload 140 | Diskip 5 (duplikat)
[OJK]    Ditemukan  48 PDF | Diupload  46 | Diskip 2 (duplikat)
[KSEI]   Ditemukan  32 PDF | Diupload  31 | Gagal  1
[Emiten] Ditemukan  18 PDF | Diupload  18

=== RINGKASAN TOTAL ===
Total Ditemukan : 243 PDF
Diupload        : 235 PDF
Diskip          : 7   (duplikat)
Gagal           : 1   (lihat collector.log)
==============================
```

Semua aktivitas dicatat ke `collector.log` untuk audit trail.

---

## Cara Menjalankan

```bash
# Jalankan semua sumber
python prospektus_collector/orchestrator.py

# Jalankan sumber tertentu saja
python prospektus_collector/orchestrator.py --source idx
python prospektus_collector/orchestrator.py --source ojk,ksei

# Retry PDF yang gagal sebelumnya
python prospektus_collector/orchestrator.py --retry-failed
```

---

## Dependencies Baru

```
firecrawl-py    # Firecrawl Python SDK
aiohttp         # Async HTTP untuk download PDF
aiofiles        # Async file I/O
boto3           # AWS SDK untuk Cloudflare R2 (S3-compatible)
```
