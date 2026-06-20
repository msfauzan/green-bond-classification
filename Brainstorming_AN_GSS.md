# Bahan Brainstorming — AN Klasifikasi Obligasi GSS Berbasis ML

> Dokumen ini ringkasan konteks untuk diajak berdiskusi. Silakan tantang asumsinya,
> usulkan alternatif, atau tunjuk titik lemahnya.

---

## 1. Konteks & Tujuan

Saya analis di **Bank Indonesia (Departemen Statistik / DSMF)**. Saya sedang menyusun
**Analytical Note (AN)** tentang kemungkinan menggunakan **machine learning** untuk
**mengklasifikasikan Efek Bersifat Utang dan Sukuk (EBUS)** di pasar Indonesia ke dalam
kategori **GSS — Green, Social, Sustainability** (dan turunannya), mengacu pada
**POJK 18/2023** (regulasi OJK tentang EBUS Berlandaskan Keberlanjutan).

Tujuan akhir: alat **klasifikasi + pemantauan pasar GSS** untuk kebutuhan statistik dan
surveilans BI.

**Singkatan:** AN = Analytical Note · EBUS = Efek Bersifat Utang & Sukuk ·
SBN = Surat Berharga Negara (obligasi pemerintah) · DJPPR = Dirjen Pengelolaan
Pembiayaan & Risiko (Kemenkeu, penerbit SBN) · OJK = otoritas pasar modal ·
SPO = Second Party Opinion · CBT = Climate Budget Tagging · BPK = Badan Pemeriksa
Keuangan · KPI/SPT = Key Performance Indicator / Sustainability Performance Target.

---

## 2. Apa yang Diklasifikasikan — POJK 18/2023 (6 bentuk)

POJK 18/2023 menetapkan **enam bentuk** EBUS Berlandaskan Keberlanjutan:

1. **EBUS Lingkungan** (Green)
2. **EBUS Sosial** (Social)
3. **EBUS Keberlanjutan** (Sustainability)
4. **EBUS Terkait Keberlanjutan** (Sustainability-Linked)
5. **Sukuk Wakaf**
6. EBUS Berwawasan Lingkungan lain yang ditetapkan OJK

---

## 3. Temuan Kunci #1 — Ini DUA Masalah Klasifikasi, Bukan Satu

Insight terpenting sejauh ini: keenam bentuk di atas **tidak** bisa diperlakukan sebagai
satu label datar, karena ada **dua mekanisme yang fundamental berbeda**:

| Tipe | Dasar penentuan status | Sinyal yang dicari |
|---|---|---|
| **Use-of-Proceeds**: Green / Social / Sustainability | **ke mana dana dipakai** (kategori proyek) | bagian *Use of Proceeds* → cocokkan ke kategori eligible |
| **Sustainability-Linked** | dana **bebas/umum**; yang menentukan adalah target kinerja | klausa **KPI + SPT + kupon step-up** |
| **Sukuk Wakaf** | struktur akad wakaf | klausa wakaf |

→ Maka skema yang diusulkan **bertingkat (hierarchical)**:

```
Level 0 — Struktur instrumen:
   ├── Use-of-Proceeds  ──► Level 1: Green / Social / Sustainability
   │                              └─ (Blue = sub-tema kelautan di bawah sini)
   ├── Sustainability-Linked   (deteksi KPI/SPT/step-up — sebagian besar rule-based)
   └── Sukuk Wakaf             (deteksi klausa wakaf)
```

Catatan: **Blue bond** bukan kelas tersendiri — ia sub-tema kelautan di bawah
Green/Sustainability (SDG 14).

---

## 4. Temuan Kunci #2 — Sovereign Sudah Otoritatif; Gap Ada di Corporate

Saya menelaah proses pemerintah (DJPPR/Kemenkeu) untuk green sukuk & SDG/Blue bond:

- Klasifikasi sovereign **bukan ML**. Prosesnya: seleksi proyek via sistem perencanaan
  anggaran (KRISNA) → **Climate Budget Tagging** → belanja **diaudit BPK** → registri
  nasional (SRN-PPI). **Labelnya sudah otoritatif & teraudit.**
- **Corporate tidak punya proses setara.** Status GSS tersebar di prospektus, *framework*
  penerbit, dan SPO yang tak terstandar. **Label resmi dari Bursa Efek hanya ~belasan
  instrumen.**

**Kesimpulan strategis:**
> Sovereign **tidak butuh ML** (sudah berlabel). ML **mendapat tempatnya di corporate** —
> di situlah gap dan kontribusi orisinalnya.

---

## 5. Keputusan Desain yang Sudah Diambil

1. **Corporate = target klasifikasi ML** (kontribusi/novelty).
   **Government = fondasi**: penyedia taksonomi, *ground truth*, dan *benchmark*.
   Alur: pakai taksonomi & label tervalidasi dari sovereign → meng-*ground* classifier →
   *deploy* ke semesta corporate yang belum terklasifikasi.

2. **Skema klasifikasi bertingkat** (lihat §3), memisahkan struktur instrumen (Level 0)
   dari use-of-proceeds (Level 1).

3. **Pendekatan teknis: taxonomy-grounded, BUKAN supervised model dari nol** (karena
   label sangat sedikit). Kombinasi:
   - *Rule-based / keyword-scoring* → baseline transparan & auditable
   - *LLM-as-judge* yang di-*ground* ke kriteria taksonomi + *few-shot* dari gold set
   - Prinsip **explainable by design** (setiap keputusan menyertakan kriteria/rujukan)

---

## 6. Realita Data (Kendala Utama)

- **Label corporate sangat sedikit (~belasan).** Terlalu kecil untuk melatih classifier
  teks dari nol → akan overfit dan tak bisa divalidasi.
- **Sumber data potensial (berlapis):**
  - *Tier 1 (sinyal terkuat):* GSS *Framework* penerbit, **SPO**, Laporan Alokasi & Dampak
  - *Tier 2:* data listing Bursa/regulator, Keterbukaan Informasi
  - *Tier 3:* Sustainability Report, Annual Report (level perusahaan — hati-hati)
- **Unit analisis = per-instrumen (ISIN), bukan per-emiten.** Sinyal terkuat melekat ke
  instrumen; laporan keberlanjutan melekat ke perusahaan — jangan dicampur.
- Bagian **paling diskriminatif = "Use of Proceeds"** → fokus ekstraksi di sini.

---

## 7. Acuan Taksonomi (Kamus Kebenaran)

Definisi "green/eligible" **tidak perlu ditemukan dari data** — sudah eksplisit di regulasi:

- **Kategori Green & Blue eligible (DJPPR):** Renewable Energy · Energy Efficiency · Green
  Tourism · Sustainable Transport · Green Buildings · Waste to Energy & Waste Management ·
  Sustainable Water & Wastewater · Sustainable Mgmt of Natural Resources (Land & Ocean) ·
  Resilience / Disaster Risk Reduction. (+ kategori Social terpisah.)
- **Standar yang konvergen (sovereign ↔ corporate):** ICMA Green/Social Bond Principles &
  Sustainability Bond Guidelines · ASEAN Bond Standards (ACMF) · POJK 18/2023 · Taksonomi
  Hijau Indonesia / TKBI.
- *Artinya kriteria green corporate & sovereign berakar pada standar yang sama — hanya
  jalur administratifnya beda. Inilah yang membuat ground truth sovereign sah dipakai
  untuk corporate.*

---

## 8. Snapshot Data Sovereign yang Sudah Saya Kompilasi

Sebagai fondasi, saya sudah menyusun lookup table **25 instrumen** GSS sovereign
(2018–2024) dari laporan resmi DJPPR. Ringkasnya:

| Kelompok | Jumlah seri | Kumulatif |
|---|---|---|
| Green Sukuk global (SNI) | 7 | USD 6,60 miliar |
| Green Sukuk ritel (ST) | 8 | IDR 40,60 T (~USD 2,78 M) |
| Green Sukuk wholesale (PBSG) | 3 | IDR 31,17 T (~USD 2,11 M) |
| **Total Green Sukuk (2018–2024)** | | **~USD 11,49 miliar** |
| SDG Bond 2024 (FRSDG001, RIEUR0932, ORI026T6) | 3 | IDR 19,15 T (~USD 1,28 M) |
| Blue Bond (Samurai, 2023 & 2024) | 4 | JPY 20,7 M + JPY 25 M |

Semua bernaung di **SDG Government Securities Framework 2021** (generasi: Green 2018 →
SDG 2021 → Thematic 2025; SPO oleh CICERO, IISD, lalu Sustainable Fitch 2025).

---

## 9. Yang Ingin Saya Brainstorming-kan

1. **Apakah skema bertingkat (§3) sudah tepat?** Ada cara lebih baik memisahkan
   sustainability-linked & wakaf dari use-of-proceeds?
2. **Strategi data-langka:** selain LLM-as-judge + rule-based, pendekatan apa lagi yang
   realistis untuk ~belasan label? (weak supervision? data programming? active learning?)
3. **Cara memperbesar gold set** secara sah — apakah harvest dari SPO & Framework penerbit
   ide yang baik, atau berisiko bias?
4. **Validasi & metrik** yang kredibel saat data uji sangat kecil dan ada kelas minoritas
   (Social/Blue/SL hampir nol contoh).
5. **Deteksi greenwashing** sebagai fitur tambahan — layak masuk scope AN atau terlalu jauh?
6. **Posisi novelty** untuk AN bank sentral: apakah framing "taxonomy-grounded
   classification untuk scaling ke semesta tak-berlabel + statistik climate finance" cukup
   kuat, atau ada angle yang lebih tajam?
7. **Risiko & blind spot** apa yang saya lewatkan?

---

*Referensi regulasi: POJK 18/2023 (OJK). Framework & laporan: DJPPR Kemenkeu (Green Sukuk
Allocation & Impact Report 2024/2025; SDG Bond & Blue Bond Report 2025). Standar: ICMA,
ASEAN Capital Markets Forum.*
