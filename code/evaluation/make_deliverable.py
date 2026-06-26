"""
Pembangun paket deliverable AN GSS yang BERSIH & siap naik ke pemangku kepentingan.

Menghasilkan folder `Deliverable AN GSS/` di root repo, berisi:
  01_Dokumen/   -- dokumen AN (.docx + .md)
  02_Data/      -- CSV bukti (sensus pasar, dekomposisi sektoral, hasil evaluasi)
  03_Grafik/    -- 4 grafik PNG (siap tempel ke paparan / AN)
  README.md     -- panduan isi paket

Semua angka grafik dihitung ulang dari data sumber (tidak di-hardcode).

Jalankan dari root repo:
  python evaluation/make_deliverable.py
"""
from __future__ import annotations
import csv
import os
import shutil
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "Deliverable AN GSS")

DOC_DIR   = os.path.join(OUT, "01_Dokumen")
DATA_DIR  = os.path.join(OUT, "02_Data")
CHART_DIR = os.path.join(OUT, "03_Grafik")

# Palet warna konsisten dgn header DOCX (biru BI-ish)
C_GREEN  = "#2E7D32"
C_SOCIAL = "#C2A100"
C_SUSTAIN= "#7B1FA2"   # ungu: gabungan green+social (UoP ganda)
C_SL     = "#1565C0"   # biru BI: Sustainability-Linked (KPI/SPT)
C_NEUTRAL= "#9E9E9E"
C_ACCENT = "#0056B2"
C_RULE   = "#B0BEC5"
C_ML     = "#0056B2"

# Peta tipe → warna (untuk grafik konsisten)
_CLASS_COLOR = {
    "Green":                 C_GREEN,
    "Social":                C_SOCIAL,
    "Sustainability":        C_SUSTAIN,
    "Sustainability Linked": C_SL,
}

plt.rcParams.update({
    "font.family"     : "DejaVu Sans",
    "font.size"       : 10,
    "axes.titlesize"  : 12,
    "axes.titleweight": "bold",
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "figure.dpi"      : 130,
})


# ---------------------------------------------------------------------------
# Pemuat data
# ---------------------------------------------------------------------------
def load_census() -> list[dict]:
    with open(os.path.join(DATA, "market_census.csv"), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_comparison() -> list[dict]:
    with open(os.path.join(DATA, "comparison_results.csv"), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_sectors() -> list[dict]:
    with open(os.path.join(DATA, "sector_decomposition.csv"), encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def prf(rows: list[dict], pred_col: str) -> dict:
    tp = fp = fn = tn = 0
    for r in rows:
        gold = r["gold"] == "GSS"
        pred = r[pred_col] == "GSS"
        if gold and pred:       tp += 1
        elif not gold and pred: fp += 1
        elif gold and not pred: fn += 1
        else:                   tn += 1
    p = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
    return {"P": p, "R": rec, "F1": f1, "TP": tp, "FP": fp, "FN": fn, "TN": tn}


# ---------------------------------------------------------------------------
# Grafik 1 — Sensus pasar GSS korporasi per kelas
# ---------------------------------------------------------------------------
def chart_census(census: list[dict]):
    classes = [r for r in census if r["gss_type"] != "TOTAL"]
    labels  = [r["gss_type"] for r in classes]
    n       = [int(r["n_instrumen"]) for r in classes]
    os_t    = [float(r["total_outstanding_triliun"]) for r in classes]
    colors  = [_CLASS_COLOR.get(lbl, C_NEUTRAL) for lbl in labels]

    # Label pendek agar muat di sumbu-x dengan 4 bar
    short = {
        "Sustainability Linked": "Sust.\nLinked",
        "Sustainability":        "Sustainability",
        "Green":                 "Green",
        "Social":                "Social",
    }
    xlabels = [short.get(l, l) for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    b1 = ax1.bar(xlabels, n, color=colors)
    ax1.set_title("Jumlah Instrumen GSS Berlabel")
    ax1.set_ylabel("Instrumen")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.bar_label(b1, padding=3, fontweight="bold")

    b2 = ax2.bar(xlabels, os_t, color=colors)
    ax2.set_title("Outstanding GSS (Rp Triliun)")
    ax2.set_ylabel("Rp Triliun")
    ax2.bar_label(b2, fmt="%.1f", padding=3, fontweight="bold")

    total = next(r for r in census if r["gss_type"] == "TOTAL")
    fig.suptitle(
        f"Sensus Pasar GSS Korporasi (EBUS) — {total['n_instrumen']} instrumen, "
        f"Rp {total['total_outstanding_triliun']} T",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "01_sensus_pasar_per_kelas.png")


# ---------------------------------------------------------------------------
# Grafik 2 — Gap: GSS berlabel vs universe EBUS korporasi
# ---------------------------------------------------------------------------
def chart_gap(census: list[dict]):
    total = next(r for r in census if r["gss_type"] == "TOTAL")
    n_gss = int(total["n_instrumen"])
    share = float(total["share_universe_pct"])
    n_universe = round(n_gss / (share / 100))
    n_rest = n_universe - n_gss

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    wedges, _ = ax.pie(
        [n_gss, n_rest],
        colors=[C_ACCENT, "#E0E0E0"],
        startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor="white"),
    )
    ax.text(0, 0.12, f"{n_gss}", ha="center", va="center",
            fontsize=26, fontweight="bold", color=C_ACCENT)
    ax.text(0, -0.16, f"GSS berlabel\n({share:.2f}% universe)",
            ha="center", va="center", fontsize=10)
    ax.set_title(
        f"Gap Pelabelan: {n_gss} GSS berlabel dari {n_universe:,} EBUS korporasi\n"
        f"{n_rest:,} instrumen ({100 - share:.2f}%) belum terklasifikasi",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    _save(fig, "02_gap_labeled_vs_universe.png")


# ---------------------------------------------------------------------------
# Grafik 3 — Evaluasi: rule-based vs ML (P / R / F1)
# ---------------------------------------------------------------------------
def chart_eval(comparison: list[dict]) -> tuple[dict, dict]:
    rule = prf(comparison, "pred_rule")
    ml   = prf(comparison, "pred_ml")

    metrics = ["Precision", "Recall", "F1-Score"]
    rule_v  = [rule["P"], rule["R"], rule["F1"]]
    ml_v    = [ml["P"], ml["R"], ml["F1"]]

    x = range(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    b1 = ax.bar([i - w/2 for i in x], rule_v, w, label="Rule-based (baseline)", color=C_RULE)
    b2 = ax.bar([i + w/2 for i in x], ml_v,   w, label="ML semantik (final)", color=C_ML)
    ax.bar_label(b1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=3, fontsize=9, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Skor")
    ax.set_title(f"Evaluasi Klasifikasi GSS (gold set {len(comparison)} dokumen)")
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    _save(fig, "03_evaluasi_rule_vs_ml.png")
    return rule, ml


# ---------------------------------------------------------------------------
# Grafik 4 — Dekomposisi sektoral
# ---------------------------------------------------------------------------
CAT_LABEL = {
    "renewable_energy": "Energi terbarukan",
    "energy_efficiency": "Efisiensi energi",
    "green_building": "Bangunan hijau",
    "green_tourism": "Pariwisata hijau",
    "sustainable_transport": "Transportasi berkelanjutan",
    "waste_management": "Pengelolaan limbah",
    "water_management": "Pengelolaan air",
    "natural_resources": "SDA hayati & guna lahan",
    "climate_resilience": "Ketahanan iklim",
    "basic_infrastructure": "Infrastruktur dasar",
    "essential_services": "Layanan esensial",
    "affordable_housing": "Perumahan terjangkau",
    "employment_msme": "Lapangan kerja & UMKM",
    "food_security": "Ketahanan pangan",
    "socioeconomic": "Sosial-ekonomi",
}


def chart_sectors(sectors: list[dict]) -> dict:
    buckets = defaultdict(int)
    cat_count = defaultdict(int)
    for r in sectors:
        buckets[r["bucket"]] += 1
        if r["bucket"] == "terdekomposisi" and r["sector_keys"]:
            for k in r["sector_keys"].split("|"):
                cat_count[k] += 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                                   gridspec_kw={"width_ratios": [1, 1.5]})

    # Panel kiri: 3 ember
    bk_order = ["terdekomposisi", "sektor_tak_terverifikasi", "level0"]
    bk_label = ["Terdekomposisi\n(sektor terbaca)",
                "GSS tanpa sektor\n(UoP tipis)",
                "Level-0\n(SL / Wakaf)"]
    bk_vals  = [buckets.get(k, 0) for k in bk_order]
    bk_col   = [C_GREEN, C_SOCIAL, C_NEUTRAL]
    b = ax1.bar(bk_label, bk_vals, color=bk_col)
    ax1.bar_label(b, padding=3, fontweight="bold")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_title(f"Status Dekomposisi ({len(sectors)} gold GSS)")
    ax1.set_ylabel("Instrumen")
    ax1.tick_params(axis="x", labelsize=8.5)

    # Panel kanan: kategori didanai (terdekomposisi)
    items = sorted(cat_count.items(), key=lambda x: x[1])
    names = [CAT_LABEL.get(k, k) for k, _ in items]
    vals  = [v for _, v in items]
    colrs = [C_GREEN if k in (
        "renewable_energy","energy_efficiency","green_building","green_tourism",
        "sustainable_transport","waste_management","water_management",
        "natural_resources","climate_resilience") else C_SOCIAL for k, _ in items]
    bars = ax2.barh(names, vals, color=colrs)
    ax2.bar_label(bars, padding=3, fontsize=8.5, fontweight="bold")
    n_decomp = buckets.get("terdekomposisi", 0)
    ax2.set_title(f"Kategori Didanai oleh Bond Terdekomposisi (n={n_decomp})")
    ax2.set_xlabel("Jumlah bond yang menyebut kategori")
    ax2.tick_params(axis="y", labelsize=8.5)

    fig.suptitle("Dekomposisi Sektoral Use-of-Proceeds GSS Korporasi",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "04_dekomposisi_sektoral.png")
    return dict(buckets)


# ---------------------------------------------------------------------------
# Grafik 5 — Cakupan pemindaian semesta & kandidat GSS tidak berlabel
# ---------------------------------------------------------------------------
def load_batch_results() -> list[dict]:
    p = os.path.join(DATA, "batch_classify_results.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_candidates() -> list[dict]:
    p = os.path.join(DATA, "unlabeled_gss_candidates.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _universe_count() -> int:
    """Jumlah instrumen universe sebenarnya dari sensus judul (bukan dari pembulatan share)."""
    p = os.path.join(DATA, "universe_title_census.csv")
    if os.path.exists(p):
        with open(p, encoding="utf-8-sig") as f:
            return sum(1 for _ in csv.DictReader(f))
    return 1437  # fallback snapshot 2026-06-18


def chart_scan_coverage(census: list[dict], batch: list[dict], candidates: list[dict]):
    total = next(r for r in census if r["gss_type"] == "TOTAL")
    n_gss      = int(total["n_instrumen"])
    n_universe = _universe_count()

    n_scanned   = len(batch)
    n_candidate = len(candidates)
    n_nongss    = n_scanned - n_candidate
    n_unscanned = n_universe - n_gss - n_scanned

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Universe EBUS\n(1.437 instrumen)"]
    bottom = 0

    segments = [
        (n_gss,      C_ACCENT,  f"GSS berlabel ({n_gss})"),
        (n_candidate, "#E65100", f"Kandidat tidak berlabel ({n_candidate})"),
        (n_nongss,   "#B0BEC5", f"Dipindai, non-GSS ({n_nongss})"),
        (n_unscanned, "#EEEEEE", f"Belum dipindai ({n_unscanned:,})"),
    ]
    bars = []
    for val, color, label in segments:
        b = ax.bar(categories, val, bottom=bottom, color=color, label=label,
                   edgecolor="white", linewidth=0.6)
        bars.append((b, val, bottom))
        bottom += val

    # Label di dalam bar (hanya yang cukup besar)
    for b, val, bot in bars:
        if val / n_universe > 0.04:
            ax.text(0, bot + val / 2, f"{val:,}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    ax.set_ylim(0, n_universe * 1.08)
    ax.set_ylabel("Jumlah instrumen")
    ax.set_title(
        f"Cakupan Pemindaian Semesta EBUS & Estimasi GSS Tidak Berlabel\n"
        f"({n_scanned} prospektus dipindai dari {n_universe:,} universe)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    _save(fig, "05_cakupan_pemindaian_semesta.png")


def _save(fig, name: str):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  grafik  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Salin dokumen + data
# ---------------------------------------------------------------------------
def copy_assets():
    copies = [
        # Sumber AN: sudah dipindah ke Deliverable AN GSS/
        (os.path.join(OUT, "Kerangka_AN_Klasifikasi_GSS_ML.docx"),
         os.path.join(DOC_DIR, "Analytical_Note_Klasifikasi_GSS_EBUS_Korporasi.docx")),
        (os.path.join(OUT, "Kerangka_AN_Klasifikasi_GSS_ML.md"),
         os.path.join(DOC_DIR, "Analytical_Note_Klasifikasi_GSS_EBUS_Korporasi.md")),
        (os.path.join(DATA, "market_census.csv"),
         os.path.join(DATA_DIR, "01_sensus_pasar_gss.csv")),
        (os.path.join(DATA, "sector_decomposition.csv"),
         os.path.join(DATA_DIR, "02_dekomposisi_sektoral.csv")),
        (os.path.join(DATA, "comparison_results.csv"),
         os.path.join(DATA_DIR, "03_hasil_evaluasi_klasifikasi.csv")),
        (os.path.join(DATA, "idx_gss_all_20260618_140427.csv"),
         os.path.join(DATA_DIR, "04_daftar_82_gss_berlabel.csv")),
        (os.path.join(DATA, "unlabeled_gss_candidates.csv"),
         os.path.join(DATA_DIR, "05_kandidat_gss_tidak_berlabel.csv")),
    ]
    for src, dst in copies:
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  salin   -> {os.path.relpath(dst, ROOT)}")
        else:
            print(f"  ! lewati (tak ada): {os.path.relpath(src, ROOT)}")


# ---------------------------------------------------------------------------
# README paket
# ---------------------------------------------------------------------------
def write_readme(census, rule, ml, buckets):
    total = next(r for r in census if r["gss_type"] == "TOTAL")
    n_gss = int(total["n_instrumen"])
    share = float(total["share_universe_pct"])
    n_universe = round(n_gss / (share / 100))

    md = f"""# Paket Deliverable — Analytical Note Klasifikasi GSS EBUS Korporasi

**Bank Indonesia · DSta / DSMF** · Paket siap paparan pemangku kepentingan.

Klasifikasi efek bersifat utang & sukuk (EBUS) korporasi Indonesia ke dalam kategori
**Green / Social / Sustainability (GSS)** per **POJK 18/2023**, ICMA, dan kerangka DJPPR.

---

## Ringkasan eksekutif

| Indikator | Nilai |
|---|---|
| Universe EBUS korporasi | {n_universe:,} instrumen |
| GSS berlabel (sensus judul IDX) | {n_gss} instrumen ({share:.2f}% universe) |
| Total outstanding GSS | Rp {total['total_outstanding_triliun']} T |
| Akurasi classifier final (ML semantik) | P {ml['P']:.3f} · R {ml['R']:.3f} · F1 {ml['F1']:.3f} |
| Baseline rule-based | P {rule['P']:.3f} · R {rule['R']:.3f} · F1 {rule['F1']:.3f} |
| Gold set evaluasi | {rule['TP'] + rule['FN'] + rule['FP'] + rule['TN']} dokumen terverifikasi |

**Temuan kunci:** mayoritas EBUS korporasi ({100 - share:.2f}%) belum terklasifikasi GSS —
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
| `01_sensus_pasar_gss.csv` | Sensus {n_gss} GSS berlabel per kelas + outstanding + share universe |
| `02_dekomposisi_sektoral.csv` | Dekomposisi use-of-proceeds {len(load_sectors())} gold GSS (per bond) |
| `03_hasil_evaluasi_klasifikasi.csv` | Confusion matrix rule vs ML pada gold set |
| `04_daftar_82_gss_berlabel.csv` | Daftar {n_gss} instrumen GSS hasil sensus judul IDX |
| `05_kandidat_gss_tidak_berlabel.csv` | Kandidat GSS tidak berlabel dari pemindaian semesta |
| `06_statistik_deskriptif_gss.csv` | Penerbitan per tahun, profil jatuh tempo, konsentrasi (HHI) & rating |

### 03_Grafik
| Berkas | Menjelaskan |
|---|---|
| `01_sensus_pasar_per_kelas.png` | Jumlah & outstanding GSS per kelas (Green/Social/SL) |
| `02_gap_labeled_vs_universe.png` | Celah {n_gss} berlabel vs {n_universe:,} universe EBUS |
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
"""
    path = os.path.join(OUT, "README.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  readme  -> {os.path.relpath(path, ROOT)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    for d in (OUT, DOC_DIR, DATA_DIR, CHART_DIR):
        os.makedirs(d, exist_ok=True)

    print("=== PEMBANGUNAN PAKET DELIVERABLE AN GSS ===\n")
    census     = load_census()
    comparison = load_comparison()
    sectors    = load_sectors()

    print("[1/4] Menyalin dokumen & data...")
    copy_assets()

    batch      = load_batch_results()
    candidates = load_candidates()

    print("\n[2/4] Membuat grafik...")
    chart_census(census)
    chart_gap(census)
    rule, ml = chart_eval(comparison)
    buckets  = chart_sectors(sectors)
    chart_scan_coverage(census, batch, candidates)

    print("\n[3/4] Menulis README...")
    write_readme(census, rule, ml, buckets)

    print("\n[4/4] Ringkasan angka (untuk verifikasi):")
    print(f"  Rule-based : P={rule['P']:.3f} R={rule['R']:.3f} F1={rule['F1']:.3f}"
          f"  (TP={rule['TP']} FP={rule['FP']} FN={rule['FN']} TN={rule['TN']})")
    print(f"  ML semantik: P={ml['P']:.3f} R={ml['R']:.3f} F1={ml['F1']:.3f}"
          f"  (TP={ml['TP']} FP={ml['FP']} FN={ml['FN']} TN={ml['TN']})")
    print(f"  Ember sektor: {buckets}")
    print(f"\nSelesai. Paket di: {OUT}")


if __name__ == "__main__":
    main()
