# Cheat Sheet — Q&A Persiapan Presentasi ke Atasan
### Klasifikasi GSS EBUS Korporasi · per 13 Juli 2026

---

## A. Angka-angka kunci (hafalkan ini)

| Metrik | Nilai |
|---|---|
| Instrumen EBUS korporasi aktif (listing IDX, scrape 13 Jul 2026) | **1.456** |
| Instrumen GSS aktif (deteksi label POJK 18/2023 dari nama) | **86 seri** (5,9% dari jumlah seri) |
| Nilai GSS outstanding | **Rp 59,0 T** (7,6% dari total EBUS aktif) |
| Emiten GSS terbesar (outstanding) | BMRI, BBRI, PNMP, SMFP, BBNI |
| Dokumen emisi terverifikasi di gold corpus | **63 dari 86 seri (73%)** |
| Kinerja mesin ML (gold set 119 dok: 53 GSS + 66 NonGSS) | **Precision 1.00 / Recall 1.00 / F1 1.00** |
| Kinerja baseline rule-based | Precision 0.92 / Recall 0.45 / F1 0.61 |
| Scan semesta 208 dokumen tak berlabel | **0 GSS tersembunyi** |

---

## B. Q&A Metodologi

**Q: Bagaimana cara menentukan sebuah obligasi itu GSS?**
A: Dua lapis. (1) *Nama instrumen* — label resmi POJK 18/2023 tercermin di nama
listing IDX ("Berwawasan Lingkungan/Sosial", "Keberlanjutan", "Terkait
Keberlanjutan"); ini deteksi paling andal. (2) *Isi prospektus* — mesin
klasifikasi membaca use-of-proceeds dan mencocokkannya ke taksonomi 9 kategori
hijau + 6 sosial (POJK 18/2023, ICMA, kerangka DJPPR).

**Q: Kenapa tidak pakai supervised machine learning biasa?**
A: Label resmi terlalu sedikit (hanya ~86 seri GSS) untuk melatih model dari
nol. Pendekatan kami *taxonomy-grounded*: aturan berbasis kata kunci bilingual
+ pencocokan semantik (sentence-transformers lokal) yang di-anchor ke taksonomi
resmi. Setiap keputusan bisa dijelaskan: kriteria/kata kunci mana yang memicu.

**Q: Model AI-nya pakai apa? Berbayar?**
A: Sepenuhnya lokal dan gratis — model MiniLM (~120 MB) berjalan di laptop,
tanpa API berbayar, tanpa data keluar. Auditable dan reproducible.

**Q: Seberapa akurat?**
A: Pada gold set 119 dokumen yang dikurasi manual (53 GSS, 66 non-GSS): mesin
ML precision 1.00, recall 1.00. Baseline rule-based hanya recall 0.45 — gagal
di dokumen Informasi Tambahan yang tipis; ini justifikasi kenapa lapisan
semantik diperlukan.

**Q: Apa jebakan klasifikasi terbesar?**
A: Kata **"Berkelanjutan"**. "Penawaran Umum Berkelanjutan (PUB)" = istilah
administratif *shelf registration*, BUKAN sustainability. Yang GSS adalah
"Keberlanjutan"/"Berwawasan". Sistem kami eksplisit memisahkan ini; contoh
nyata: prospektus FIFA "Obligasi Berkelanjutan VII" terdeteksi benar sebagai
obligasi biasa.

**Q: Nilai tambahnya apa? Bukannya dari nama saja sudah ketahuan GSS?**
A: Deteksi biner memang murah. Nilai tambahnya: (1) **dekomposisi sektoral** —
use-of-proceeds mengalir ke kategori eligible mana (energi terbarukan,
transportasi bersih, UMKM, dst.) → statistik yang belum ada; (2) **verifikasi
klaim** — screening greenwashing dengan membandingkan proceeds vs taksonomi;
(3) **kesenjangan data korporasi vs sovereign** — SBN sudah punya proses
otoritatif (CBT→BPK→SRN-PPI), korporasi belum; ini kontribusi orisinalnya.

---

## C. Q&A Data & Dokumen

**Q: Dari mana datanya?**
A: Listing resmi IDX (scrape 13 Jul 2026, 1.456 instrumen) + dokumen emisi dari
pengumuman IDX (255 pengumuman, ±590 file, 3 kata kunci pencarian) + 25
instrumen SBN sebagai ground truth taksonomi.

**Q: Kenapa dokumen hanya lengkap 73% (63/86)?**
A: 23 seri sisanya **tidak tersedia di pengumuman IDX** — sudah dipastikan
lewat 3 kata kunci pencarian dan OCR terhadap 100+ dokumen scan. Perlu sumber
alternatif: e-BOCS OJK atau situs emiten. Di dashboard, baris ini berlabel
"❌ Tidak ada di IDX" (bukan hilang — jadi checklist QC).

**Q: Kenapa banyak dokumen "Informasi Tambahan Ringkas", bukan prospektus?**
A: Karena mayoritas emisi GSS memakai skema **PUB (shelf registration)**:
prospektus penuh hanya terbit di Tahap I; tahap lanjutan secara hukum memang
hanya menerbitkan *Informasi Tambahan*. Jadi itu dokumen resmi yang benar,
bukan dokumen yang salah. Komposisi korpus: 53 Informasi Tambahan Ringkas,
4 prospektus penuh, 4 iklan koran, 3 prospektus ringkas — jenis dideteksi
dari ISI dokumen, bukan nama file.

**Q: Bagaimana kualitas korpus dijaga?**
A: Gate kurasi otomatis mensyaratkan sampul dokumen cocok dengan instrumen di
listing IDX pada 5 atribut (penanda GSS, tahun, tahap, seri, jenis instrumen);
surat pengantar dan laporan pemeringkatan otomatis ditolak; per instrumen
dipilih dokumen paling substantif. Di atas itu tetap ada QC manual — kolom
"Dokumen" di dashboard dibuat untuk itu.

**Q: Data per kapan? Bisa di-update?**
A: Listing 13 Juli 2026. Update = jalankan ulang scraper (path otomatis ke
hasil terbaru) → kurasi → rebuild. Satu siklus < 1 jam, semua otomatis.

---

## D. Q&A Angka pasar (kalau ditanya detail)

**Q: Komposisi 86 seri GSS aktif?**
A: Green 22, Social 16, Sustainability 11, Sustainability-Linked 4 (dari 53
yang terverifikasi dokumen; distribusi penuh lihat dashboard). Social
mendominasi nilai outstanding (didorong PNM/Pegadaian/SMF — sosial-orange &
UMKM).

**Q: Porsi GSS kok kecil (7,6%)?**
A: Konsisten dengan pasar: EBUS korporasi GSS Indonesia masih tahap awal.
Justru itu argumen AN ini — perlu infrastruktur pengukuran sebelum pasarnya
membesar. Porsi turun tipis dari 7,7% (Juni) karena pasar total tumbuh lebih
cepat dari segmen GSS.

**Q: Ada GSS yang "tersembunyi" (tidak berlabel tapi sebenarnya GSS)?**
A: Sudah discan: 208 dokumen emiten tak berlabel → 0 kandidat GSS. Gap
labeling di Indonesia adalah gap penerbitan, bukan gap deteksi.

---

## E. Kelemahan yang harus diakui duluan (jangan sampai ditanya baru ngaku)

1. **23/86 seri belum ada dokumennya** — bukan kegagalan metode, tapi
   keterbatasan kanal publikasi IDX; tindak lanjut manual sudah teridentifikasi.
2. **Recall rule-based rendah (0.45)** — disengaja ditampilkan sebagai
   baseline; naratifnya "transparan tapi kurang sensitif → dilengkapi ML".
3. **ML P/R 1.00 di 119 dokumen** — angka sempurna di gold set kecil; jangan
   overclaim, framing yang aman: "pada korpus terverifikasi saat ini".
4. **Sektor decomposition bergantung kualitas teks PDF** — dokumen scan butuh
   OCR; sebagian iklan koran tipis informasinya.
5. **Ground truth sub-kelas** dari nama instrumen + kurasi manual, belum
   divalidasi pihak ketiga (SPO/external review belum diolah).

---

*Dashboard: `streamlit run code/webapp/app.py` → localhost:8501. Semua kode &
data kecil di GitHub `msfauzan/green-bond-classification` (main). PDF korpus
QC di OneDrive `GSS Prospektus QC/`.*
