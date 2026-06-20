# Cleanup Report — 2026-06-17

Pembersihan dan deduplikasi direktori proyek **Green Bond Classification**.
Tujuan: menghilangkan redundansi (file duplikat, scratch, cache) tanpa
menghapus file penting dan tanpa mengubah kode aplikasi.

## Ringkasan dampak

| Metrik | Sebelum | Sesudah |
|--------|---------|---------|
| Ukuran proyek (tanpa `.git`/`.worktrees`) | ~1.56 GB | ~827 MB |
| Ruang dibebaskan | — | **~739 MB** |
| File git ter-track yang terpengaruh | — | **0** (tidak ada) |

Seluruh item yang dihapus berstatus **gitignored atau untracked** — tidak ada
satu pun file yang dikelola git yang ikut terhapus. `git status` setelah
pembersihan identik dengan sebelum pembersihan.

## 1. Data duplikat (~739 MB)

`pdf_by_content/` dan `Prospektus_Downloaded/` ternyata menyimpan PDF yang
**byte-identical** (sama persis), hanya beda cara pengorganisasian:

- `pdf_by_content/` — diorganisir per **jenis konten** (01_prospektus_utama, dst.)
- `Prospektus_Downloaded/` — diorganisir per **tanggal/emiten**

Verifikasi: 730 hash konten unik di kedua direktori; pengecekan MD5 memastikan
**0 konten** di `Prospektus_Downloaded/` yang tidak ada di `pdf_by_content/`.

**Tindakan:**
- Disepakati untuk **mempertahankan `pdf_by_content/`** dan menghapus `Prospektus_Downloaded/`.
- Sebelum penghapusan, **35 file yang hanya ada di `Prospektus_Downloaded/`**
  diselamatkan ke folder baru:
  `pdf_by_content/09_dari_prospektus_downloaded/` (~14 MB).
  File ber-nama kurung (mis. `Penyampaian Prospektus [ADMF ].pdf`) diberi prefiks
  nama folder asal agar tidak bentrok.
- `Prospektus_Downloaded/` dihapus (739 MB) setelah verifikasi nol kehilangan data.

> Catatan: `pdf_by_content/` dapat di-regenerasi dari sumber via
> [scripts/categorize_by_content.py](../scripts/categorize_by_content.py).

## 2. File scratch / output di root (dihapus)

Semuanya gitignored atau untracked, merupakan hasil eksperimen / output sementara:

| File | Keterangan |
|------|-----------|
| `scrape_idx.py`, `scrape_full.py` | Skrip Playwright one-off scraping IDX (untracked) |
| `test_api.py` (root) | Skrip scratch Playwright — **bukan** test. Test asli ada di [tests/test_api.py](../tests/test_api.py) |
| `html-idx.html`, `idx_bonds_table_0.html` | HTML scratch |
| `hasil_klasifikasi_parallel.xlsx`, `hasil_klasifikasi_realtime.xlsx` | Output klasifikasi (regenerable) |
| `download_log.xlsx` | Log unduhan |
| `install.cmd` | Skrip instalasi sementara |

Root sekarang hanya berisi: `README.md`, `requirements.txt`,
`Tabel Referensi AN Green Bond.xlsx`, dan direktori-direktori proyek.

## 3. Cache & temporary (dibersihkan)

Semua di-regenerasi otomatis:

- Seluruh `__pycache__/` (di luar `.git/` dan `.worktrees/`)
- `.pytest_cache/`
- `.serena/cache/` (memori `.serena/memories/` & konfigurasi **tetap dipertahankan**)
- `.playwright-mcp/*.log`

## Yang TIDAK diubah (sengaja dipertahankan)

- **Kode aplikasi** — tidak ada satu baris pun yang diubah.
- **Git worktree** `.worktrees/prospektus-collector` — branch `feature/prospektus-collector`
  belum di-merge ke `main`, jadi tidak disentuh.
- **Perubahan git yang sedang berjalan** (refactor pemindahan skrip ke `scripts/`) —
  dibiarkan apa adanya untuk Anda commit sendiri.
- `api/requirements.txt` & `webapp/requirements.txt` — bukan duplikat; keduanya
  hanya `-r ../requirements.txt` (kebutuhan konteks deployment).
- `ML_Dataset/` (varian CSV), `prospektus_by_category/`, `klasifikasi prospektus/` —
  dataset/organisasi berbeda yang berukuran kecil; dipertahankan.

## Verifikasi pasca-pembersihan

- ✅ Sintaks seluruh modul inti valid (`classifier/*`, `webapp/api.py`, `api/index.py`).
- ✅ Semua direktori penting masih ada.
- ✅ `git status` tidak berubah (tidak ada file track yang terhapus).
- ⚠️ Suite `pytest` tidak dijalankan: environment ini belum memiliki virtualenv/pytest
  terpasang (kondisi pra-existing, di luar lingkup pembersihan). Untuk menjalankan:
  `python -m venv .venv && .venv\Scripts\pip install -r requirements.txt pytest && .venv\Scripts\pytest -q`.
