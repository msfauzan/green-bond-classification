# Ringkasan Verifikasi — Dokumen Emisi GSS yang Tidak Ditemukan di IDX
### per 13 Juli 2026 · korpus final 72/86 seri aktif (84%)

## Metode pencarian (exhaustive)

Seluruh pengumuman IDX (menu Berita → Pengumuman, API `GetAllAnnouncement`)
ditelusuri untuk emiten terkait dengan **5 kata kunci**: "prospektus",
"informasi tambahan", "fakta material", "keberlanjutan", "berwawasan" —
total 500+ pengumuman, 1.100+ file diunduh. Dokumen scan (100+ file) di-OCR
(RapidOCR lokal) sebelum dinyatakan tidak relevan. Pencocokan dokumen →
instrumen memakai 5 atribut nama (penanda GSS, tahun, tahap, seri, jenis
instrumen) terhadap teks sampul.

## 14 seri yang dokumen emisinya TIDAK tersedia di pengumuman IDX

| Emiten | Instrumen (seri) | Bukti ketidaktersediaan |
|---|---|---|
| BMRI | Obligasi Keberlanjutan Berkelanjutan I Tahap I 2025 (3 seri) | Pengumuman BMRI Mar 2025 hanya memuat ITR *Obligasi Berwawasan Lingkungan* Tahap II 2025 (emisi green yang berbeda, sudah ter-cover). Dokumen emisi Keberlanjutan-nya tidak dipublikasikan via pengumuman IDX. |
| ISSP | Obligasi Terkait Keberlanjutan I SPINDO 2024 (Seri A, B, C) | 24+ dokumen ISSP diperiksa (2023–2026): semuanya emisi lain/laporan; 4 scan di-OCR tanpa hasil. Dokumen SLB 2024 tidak ada di pengumuman. |
| PNMP | Obligasi Berwawasan Sosial Orange Berkelanjutan I Tahap I 2025 (3 seri) | Pengumuman PNM 2025–2026 hanya memuat ITR *Sukuk Mudharabah* Sosial Orange (kembarannya sukuk ter-cover); dokumen sisi *obligasi*-nya tidak dipublikasikan terpisah di IDX. |
| PPGD | Obligasi + Sukuk Mudharabah Berwawasan Sosial Berkelanjutan II Tahap I 2026 (4 seri) | Pengumuman Pegadaian 2026 hanya laporan tahunan/ESG; dokumen emisi PUB II Tahap I 2026 belum muncul di pengumuman IDX. |
| SMII | Obligasi Keberlanjutan Berkelanjutan I Tahap I 2025 (1 seri) | Nama instrumen hanya muncul di **tabel daftar utang** dalam ITR PUB IV (Nov 2025) — bukan dokumen emisinya (jebakan konteks neraca, sengaja tidak diterima matcher). Dokumen emisi Tahap I 2025-nya sendiri tidak ada. |

## Tindak lanjut yang disarankan

1. **e-BOCS OJK** (sistem penyampaian dokumen emisi) — kanal resmi yang lebih
   lengkap daripada pengumuman IDX.
2. **Situs emiten** (bagian hubungan investor) — BMRI, PNM, dan Pegadaian
   biasanya memuat prospektus/IT di situsnya.
3. Baris-baris ini berlabel **"❌ Tidak ada di IDX"** di dashboard (tabel
   Daftar Instrumen GSS) sebagai checklist QC.

## Temuan proses yang penting

- **Bug scraper (sudah diperbaiki)**: pemotongan nama folder 75 karakter bisa
  menyisakan spasi di ujung → Windows menolak path → lampiran gagal tersimpan
  secara diam-diam. Ini penyebab dokumen FIFA "hilang"; setelah diperbaiki,
  FIFA Keberlanjutan Orange langsung ketemu dan ter-cover.
- Dokumen tahap lanjutan PUB secara hukum berupa *Informasi Tambahan* (bukan
  prospektus penuh) — komposisi 83 baris gold DB final: 64 Info Tambahan
  Ringkas, 9 Iklan Ringkas, 4 Prospektus penuh, 3 Prospektus Ringkas,
  3 Dokumen Emisi lain (72 seri aktif + 11 seri jatuh tempo).
