# Lookup Table SBN GSS — Catatan

Sumber:
- **Green Sukuk** → 2024 & 2025 Green Sukuk Allocation & Impact Report (emisi 2018–2024).
- **SDG Bond & Blue Bond** → 2025 SDG Bond and Blue Bond Allocation & Impact Report (emisi 2024).
- Blue Bond 2023 → siaran pers DJPPR/UNDP.

PDF sumber tersimpan di [laporan_djppr/](laporan_djppr/):
`Green_Sukuk_..._2024.pdf`, `..._2025.pdf`, `SDG_Bond_Allocation_Impact_Report.pdf`.

File data: [sbn_gss_lookup.csv](sbn_gss_lookup.csv)

## Total kumulatif (s.d. 2024 — dari Report 2025)

| Kanal | Kumulatif |
|---|---|
| Green Sukuk **global (SNI)** | USD 6,60 miliar |
| Green Sukuk **ritel (ST)** | IDR 40,60 triliun (~USD 2,78 miliar) |
| Green Sukuk **wholesale (PBSG 001)** | IDR 31,17 triliun (~USD 2,11 miliar) |
| **Total Green Sukuk** | **~USD 11,49 miliar** |
| Emisi FY2024 saja | USD 600 jt + IDR 9,92 T (ST) + IDR 8,95 T (PBSG) |

Pembanding s.d. 2023 (Report 2024): total **USD 9,59 miliar**.

## SDG Bond & Blue Bond (emisi FY2024)

| Instrumen | Total 2024 |
|---|---|
| **SDG Bond** (FRSDG001 + RIEUR0932 + ORI026T6) | IDR 19,15 T (~USD 1,28 M) |
| **Blue Bond** (RIJPY0531B/0534B/0544, Samurai) | JPY 25 M (~IDR 2,56 T) |

SDG Bond = tema **Sosial+Hijau** (mendanai PKH, pelatihan vokasi, dll → Sustainability).
Blue Bond 2024 = Samurai 3 tranche (7/10/20 thn), terkait utama **SDG 14** (kelautan).
Catatan: kode `RIEUR0932`, `ORI026T6`, `RIJPY...` adalah **kode seri**, bukan ISIN
penuh — ISIN proper perlu konfirmasi BI-SSSS/KSEI/Euroclear.

## Kategori proyek eligible (use of proceeds)

Renewable Energy · Energy Efficiency · Sustainable Transport · Sustainable
Management of Natural Resources on Land · Green Building · Sustainable Water &
Wastewater Management · Resilience to Climate Change for Highly Vulnerable
Areas / Disaster Risk Reduction.

Contoh breakdown sektor (PBSG 001, FY2023): Sustainable Transport ~82,8% ·
Resilience ~16,5% · sisanya kecil. Split mitigasi/adaptasi historis ~57/43.

## ISIN yang sudah terkonfirmasi (dari Report 2024 & 2025)

| Seri | ISIN |
|---|---|
| SNI seri-6 (2023) | `US71567RAY27` |
| SNI 0754 seri-7 (2024) | `USY68613AC56` |
| ST-012T4 (2024) | `IDJ000030903` |
| ST-013T4 (2024) | `IDJ000033709` |
| PBSG 001 emisi-3 (2024) | `IDP000005308` |

## Yang masih perlu diverifikasi (jangan dianggap final)

- **ISIN seri 2018–2022** (SNI seri-1…5, ST-006…011, PBSG emisi-1/2) belum ada di
  laporan — tarik dari data internal BI (BI-SSSS) / KSEI / Bloomberg.
- **Kode seri ritel ST-006…ST-009** — *inferensi* dari kalender emisi (cocok by
  tanggal), bukan tertulis eksplisit. ST-010T4…ST-013T4 eksplisit di laporan.
- **PBSG 001 emisi-3 (2024)** — transaction summary tertulis 4thn/6,55%, tetapi
  infografis milestone tertulis 5thn/6,625%. **Diskrepansi di sumber** — konfirmasi DJPPR.
- **Blue bond** — nominal/kupon dari siaran pers; konfirmasi ISIN & seri ke DJPPR.
- Laporan berhenti di **FY2024**. Emisi 2025 belum termasuk — update dari siaran
  pers DJPPR / Report edisi berikutnya.

## Catatan taksonomi

Kolom `kategori_gss` = klasifikasi POJK-style (Green/Social/Sustainability);
kolom `tema` memisahkan **Blue** sebagai sub-tema kelautan di bawah payung
Sustainability/Green — bukan kelas keempat. SDG Government Securities Framework
(2021) adalah payung yang mengintegrasikan Green + Social + Blue.
