# IDX Obligasi & Sukuk Korporasi Scraper

Scraper data **Obligasi & Sukuk Korporasi** dari website IDX (Bursa Efek Indonesia):
<https://www.idx.co.id/id/data-pasar/obligasi-sukuk/obligasi-sukuk-korporasi/>

## Kenapa pakai browser (Selenium)?

Website IDX dilindungi **Cloudflare** (bot detection) + TLS fingerprinting. `requests`,
`curl`, atau `httpx` biasa **pasti diblok** (HTTP 403 / koneksi putus). Solusinya:
menjalankan **Chrome sungguhan** lewat Selenium supaya lolos challenge, lalu menarik
data dari endpoint JSON internal IDX langsung dari dalam konteks browser (mewarisi
cookie clearance Cloudflare).

Alur script:
1. Buka halaman pakai Chrome → tunggu Cloudflare lolos.
2. **Auto-discovery**: baca network log browser untuk menemukan endpoint API JSON
   yang dipanggil halaman — jadi tidak bergantung pada URL yang di-hardcode.
   Endpoint yang ditemukan saat ini:
   `https://www.idx.co.id/secondary/get/BondSukuk/bond?pageSize=10&indexFrom=1&bondType=1`
   (`bondType=1` = korporasi; `indexFrom` = nomor halaman 1-based).
3. Tarik **semua** record via `fetch()` di dalam browser. ⚠️ API IDX
   mengembalikan baris dalam urutan **tidak stabil**, jadi paginasi per-halaman bisa
   *melewatkan* record (mis. emiten ARKO sempat hilang). Karena itu script mengambil
   **semua sekaligus dalam satu request** (`pageSize` = total, server melaporkan
   `ResultCount` ± 1.437 seri). Paginasi+dedup hanya dipakai sebagai cadangan.
4. Bersihkan padding spasi → simpan ke `../data/` sebagai **CSV + JSON**.
5. (opsional) Filter hanya emiten pada `list_emisi_gss.txt`.

## Instalasi

```bash
pip install -r requirements.txt
```

Butuh **Google Chrome** terpasang. `chromedriver` diurus otomatis oleh Selenium Manager
(bawaan Selenium 4) — tidak perlu download manual.

## Cara pakai

```bash
# Tarik semua obligasi/sukuk korporasi -> data/idx_obligasi_sukuk_korporasi_<timestamp>.csv
python scrape_idx_bonds.py

# Sekaligus buat subset hanya emiten GSS (green/social/sustainability) dari list
python scrape_idx_bonds.py --filter ../list_emisi_gss.txt

# Opsi lain
python scrape_idx_bonds.py --headless     # tanpa jendela (kadang kena Cloudflare)
python scrape_idx_bonds.py --uc           # coba undetected-chromedriver dulu (CF ketat)
python scrape_idx_bonds.py --debug        # simpan sample payload mentah utk inspeksi
python scrape_idx_bonds.py --url "<API>"  # paksa endpoint API tertentu
```

> Jalankan **dengan jendela terlihat** (default) untuk peluang terbaik lolos Cloudflare.
> Kalau gagal di percobaan pertama, ulangi — challenge kadang butuh 2x.

## Output

File di folder `data/` (relatif terhadap root repo):

| File | Isi |
|------|-----|
| `idx_obligasi_sukuk_korporasi_<ts>.csv` / `.json` | semua seri |
| `idx_obligasi_sukuk_korporasi_gss_<ts>.csv` / `.json` | hanya emiten di file `--filter` |

Kolom: `Nomor, BondId, BondName, IssuerCode, MatureDate, Rating, Outstanding`.

## Catatan

- Data IDX hanya untuk keperluan non-komersial sesuai ketentuan PT Bursa Efek Indonesia.
- `undetected-chromedriver` bersifat opsional (butuh versi chromedriver yang cocok dengan
  Chrome). Default script pakai Selenium standar yang sudah cukup melewati Cloudflare.
