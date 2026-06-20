# Kerangka Analytical Note (AN)
## Klasifikasi Efek Bersifat Utang & Sukuk GSS Berbasis Machine Learning

**Unit:** Bank Indonesia — DSta / DSMF
**Status:** Draft kerangka · 2026-06-17
**Dokumen terkait:** [data/sbn_gss_lookup.csv](data/sbn_gss_lookup.csv) · [data/sbn_gss_lookup_NOTES.md](data/sbn_gss_lookup_NOTES.md) · [data/laporan_djppr/](data/laporan_djppr/)

---

## 0. Ringkasan Eksekutif

AN ini mengusulkan pendekatan **klasifikasi GSS (Green, Social, Sustainability) yang
di-*ground* ke taksonomi resmi** untuk mengidentifikasi dan memantau Efek Bersifat
Utang dan Sukuk berwawasan lingkungan/sosial/keberlanjutan di pasar Indonesia.

Posisi inti: **klasifikasi sovereign sudah otoritatif** (proses Climate Budget
Tagging → audit BPK → registri SRN-PPI), sehingga **tidak memerlukan ML**. Gap nyata
ada di **sisi corporate**, yang tidak memiliki proses penandaan setara dan hanya
memiliki segelintir instrumen berlabel. Karena itu:

> **Corporate = objek klasifikasi ML (kontribusi orisinal).
> Government = fondasi taksonomi, *ground truth*, dan *benchmark*.**

---

## 1. Latar Belakang & Rumusan Masalah

- Pasar GSS Indonesia berkembang pesat: green sukuk sovereign kumulatif **~USD 11,49
  miliar (2018–2024)**, ditambah SDG Bond dan Blue Bond. (Lihat lookup table.)
- Untuk **sovereign**, status GSS jelas dan tervalidasi: setiap emisi mengikuti
  *Framework* resmi, ditandai via **Climate Budget Tagging (CBT)**, belanjanya
  **diaudit BPK**, lalu didaftarkan ke **SRN-PPI**.
- Untuk **corporate**, tidak ada proses setara. Klasifikasi tersebar di prospektus,
  *framework* penerbit, dan SPO yang tidak terstandar; **label resmi BEI hanya ~belasan
  instrumen.** Akibatnya, semesta GSS corporate **tidak terpetakan utuh** untuk
  kebutuhan statistik dan surveilans.

**Masalah inti:** Bagaimana mengklasifikasikan dan memantau status GSS pada semesta
penerbit **corporate** yang tidak melewati proses penandaan resmi, secara konsisten
terhadap taksonomi POJK 18/2023 dan standar internasional, meskipun data berlabel
sangat terbatas?

---

## 2. Gap & Posisi terhadap Proses yang Ada

| Dimensi | Proses DJPPR (sovereign) | Kebutuhan AN ini (corporate) |
|---|---|---|
| Penentuan status GSS | KRISNA + CBT + audit BPK | Tidak ada → **gap** |
| Cakupan | Hanya instrumen pemerintah | Seluruh penerbit corporate |
| Sifat | Manual, administratif, terverifikasi | Perlu **otomatis & skalabel** |
| Output | Label per emisi (otoritatif) | Klasifikasi + skor keyakinan + flag |

**Novelty AN:** bukan menggantikan proses resmi, melainkan **men-*scale* taksonomi
resmi ke wilayah yang belum tertandai** + skrining indikasi *greenwashing*. Pendekatan
ini sejalan dengan literatur terkini (lihat §9: kompilasi statistik climate finance
*security-by-security*, dan AI untuk konsistensi pengungkapan transisi).

---

## 3. Pertanyaan & Tujuan AN

**Pertanyaan penelitian**
1. Dapatkah taksonomi GSS resmi (Framework DJPPR + POJK 18/2023, berakar ICMA/ASEAN)
   dipakai sebagai *ground truth* untuk mengklasifikasi EBUS corporate secara otomatis?
2. Seberapa andal klasifikasi tersebut diukur terhadap label gold yang tersedia?
3. Seberapa besar selisih antara semesta GSS *berlabel resmi* dan *hasil klasifikasi*
   — yakni potensi pasar GSS yang belum terpetakan?

**Tujuan**
- Membangun *pipeline* klasifikasi GSS corporate yang di-*ground* ke taksonomi resmi.
- Menyediakan basis **statistik pasar GSS** yang lebih lengkap untuk BI.
- Menyediakan **alat skrining awal** indikasi *greenwashing* / inkonsistensi.

---

## 4. Ruang Lingkup & Pembagian Peran

| Segmen | Peran dalam AN | Alasan |
|---|---|---|
| **Government (SBN)** | Fondasi: taksonomi, *ground truth*, *benchmark* | Sudah berlabel & teraudit; ML tidak relevan |
| **Corporate (EBUS)** | Target klasifikasi ML | Tidak ada proses resmi; label langka → gap |

**Alur logis:**
> Taksonomi & contoh tervalidasi (sovereign) → meng-*ground* classifier →
> *deploy* ke semesta corporate (belum terklasifikasi) → statistik & surveilans BI.

Kelas keluaran: **Green · Social · Sustainability**, dengan **Blue** sebagai sub-tema
kelautan di bawah Sustainability/Green (bukan kelas keempat) — konsisten dengan
struktur SDG Government Securities Framework.

---

## 5. Kerangka Sumber Data (berlapis)

**Tier 1 — sinyal terkuat (status GSS eksplisit)**
- GSS Bond *Framework* penerbit (4 komponen inti POJK 18/2023)
- *Second Party Opinion* (SPO)
- Laporan Alokasi Dana & Dampak

**Tier 2 — registry/regulator (sebagian berlabel)**
- Data listing BEI/KSEI/OJK; Keterbukaan Informasi IDX
- **Taksonomi acuan** (POJK 18/2023, TKBI/Taksonomi Hijau) — basis kriteria

**Tier 3 — konteks penerbit (pelengkap)**
- *Sustainability Report* (POJK 51/2017), Annual Report, rating ESG

**Sisi sovereign (fondasi, sudah dikompilasi):**
- 2024 & 2025 Green Sukuk Allocation & Impact Report
- 2025 SDG Bond & Blue Bond Allocation & Impact Report
- → [data/sbn_gss_lookup.csv](data/sbn_gss_lookup.csv) (25 instrumen)

> **Unit analisis = per-instrumen (ISIN), bukan per-emiten.** Sinyal terkuat melekat
> ke instrumen; *Sustainability Report* melekat ke perusahaan — jangan dicampur.

---

## 6. Taksonomi Acuan (kamus kebenaran)

**Kategori Green & Blue eligible (acuan DJPPR/Framework):**
Renewable Energy · Energy Efficiency · Green Tourism · Sustainable Transport ·
Green Buildings · Waste to Energy & Waste Management · Sustainable Water & Wastewater ·
Sustainable Management of Natural Resources (Land & Ocean) · Resilience / Disaster
Risk Reduction. (+ kategori Social terpisah.)

**Standar yang dirujuk (konvergen sovereign ↔ corporate):**
- **ICMA** — Green/Social Bond Principles, Sustainability Bond Guidelines
- **ACMF** — ASEAN Green/Social/Sustainability Bond Standards
- **Nasional** — POJK 18/2023, Taksonomi Hijau Indonesia / TKBI
- Impact: ICMA Harmonised Framework, UNDP SDG Impact Standards

*(Catatan: kriteria green corporate dan sovereign berakar pada standar yang sama —
hanya jalur administratifnya berbeda. Ini yang membuat ground truth sovereign valid
dipakai untuk corporate.)*

---

## 7. Metodologi

Mengingat **label gold kecil**, pendekatan **bukan** supervised model dari nol,
melainkan **taxonomy-grounded + LLM/transfer learning**, berlapis:

**Tahap 1 — Ekstraksi**
- Ambil bagian **Penggunaan Dana / Use of Proceeds** (bagian paling diskriminatif)
  dari prospektus/framework/SPO. Ini *leverage* terbesar.

**Tahap 2 — Klasifikasi (hybrid)**
- **Rule-based / keyword-scoring** terhadap 9 kategori eligible → baseline transparan
  & auditable (penting untuk konteks BI). *(Embrio sudah ada di repo: `classifier/`.)*
- **LLM-as-judge** di-*ground* ke kriteria taksonomi (§6) + *few-shot* dari gold set →
  menangani variasi bahasa & semantik yang luput dari keyword.
- Output: kelas GSS + **skor keyakinan** + rujukan kriteria yang terpenuhi.

**Tahap 3 — Verifikasi / skrining greenwashing**
- Bandingkan klaim penerbit vs kriteria terpenuhi → *flag* inkonsistensi.

**Tahap 4 — Agregasi statistik**
- Kompilasi *security-by-security* → statistik pasar GSS untuk BI.

**Prinsip:** *explainable by design* — setiap keputusan menyertakan kriteria/rujukan,
bukan kotak hitam. Selaras kebutuhan pertanggungjawaban kebijakan.

---

## 8. Evaluasi & Validasi

- **Gold set** = ~belasan label BEI (corporate) + 25 instrumen sovereign tervalidasi
  + (perluasan) label dari Framework/SPO yang tersedia.
- **Metrik:** precision/recall/F1 per kelas; perhatikan **kelas minoritas**
  (Social/Blue) yang contohnya sedikit.
- **Validasi silang** rule-based vs LLM-judge → ukur kesepakatan; tinjau manual yang
  berbeda.
- **Benchmark eksternal** (opsional): Climate Bonds Initiative, Bloomberg/Refinitiv
  green flag.
- Laporkan **keterbatasan data secara eksplisit** — menambah kredibilitas AN.

---

## 9. Relevansi Mandat BI / DSta

- **Statistik:** kompilasi climate finance *security-by-security* — memetakan pasar
  GSS yang belum tertangkap label resmi (sejalan praktik bank sentral terkini).
- **Surveilans / stabilitas:** skrining *greenwashing* & inkonsistensi pengungkapan.
- **Kebijakan keuangan berkelanjutan:** basis data untuk pendalaman pasar EBUS GSS.

Selaras paper referensi proyek: kompilasi statistik climate finance *s-b-s*, dan AI
untuk menilai konsistensi pengungkapan transisi (lihat
[Paper Referensi ML/](Paper%20Referensi%20ML/)).

---

## 10. Keterbatasan & Risiko (jujur)

- **Data berlabel kecil** → andalkan LLM/transfer learning, bukan model besar; akurasi
  perlu dilaporkan apa adanya, terutama kelas minoritas.
- **Risiko pseudo-labeling:** label benih bias menyebar; jangan jadikan tulang punggung
  sebelum *pipeline* utama stabil.
- **Kualitas dokumen:** prospektus panjang & tak terstruktur; ekstraksi *use-of-proceeds*
  bisa gagal → perlu QA.
- **Level analisis:** jangan mencampur sinyal level-instrumen vs level-perusahaan.
- **Pembaruan:** taksonomi & Framework berevolusi (2018 → 2021 → 2025) — acuan harus
  diberi versi.

---

## 11. Tahapan Kerja (indikatif)

1. **Fondasi** — finalisasi lookup sovereign + lembar taksonomi acuan (*sebagian besar
   selesai*).
2. **Gold set corporate** — kompilasi label BEI + harvest Framework/SPO; nilai
   kecukupannya.
3. **Pipeline v1** — ekstraksi *use-of-proceeds* + classifier hybrid (rule + LLM-judge).
4. **Evaluasi** — metrik terhadap gold set; tinjau kasus sulit.
5. **Agregasi** — statistik pasar GSS + estimasi *gap* berlabel vs terklasifikasi.
6. **Penulisan AN** — temuan, keterbatasan, implikasi kebijakan.

---

## 12. Referensi Utama

- Republic of Indonesia — Green Bond & Green Sukuk Framework (2018); SDG Government
  Securities Framework (2021); Thematic Bonds & Sukuk Framework (2025).
- DJPPR Kemenkeu — Green Sukuk Allocation & Impact Report 2024, 2025; SDG Bond & Blue
  Bond Allocation & Impact Report 2025.
- POJK 18/2023; Taksonomi Hijau Indonesia / TKBI.
- ICMA GBP/SBP/SBG; ACMF ASEAN GBS/SBS/SUS.
- SPO: CICERO (2018, 2021), CICERO & IISD (2021), Sustainable Fitch (2025).
- Paper referensi proyek (security-by-security climate finance; AI transition
  disclosure) — [Paper Referensi ML/](Paper%20Referensi%20ML/).
