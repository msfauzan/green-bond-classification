"""
Statistik deskriptif pasar GSS korporasi — pelengkap sensus di make_deliverable.py.

Menghasilkan 3 grafik struktural + 1 CSV agregat, langsung ke folder deliverable:
  03_Grafik/06_tren_penerbitan_gss.png      -- penerbitan per tahun (jumlah + outstanding)
  03_Grafik/07_profil_jatuh_tempo.png       -- outstanding jatuh tempo per tahun (maturity ladder)
  03_Grafik/08_konsentrasi_dan_rating.png   -- konsentrasi penerbit (HHI) + sebaran rating
  02_Data/06_statistik_deskriptif_gss.csv   -- semua angka grafik (audit trail)

Sumber: data/universe_title_census.csv (82 GSS-titled dari 1.437 universe).
Tahun penerbitan diekstrak dari nama instrumen ("... Tahun YYYY ..."); seluruh 82
instrumen memuat penanda tahun ini (0 gagal-parse — divalidasi di _self_check).

Jalankan dari root repo:
  python code/evaluation/descriptive_stats.py
"""
from __future__ import annotations
import csv
import os
import re
from collections import defaultdict, Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
DATA = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "Deliverable AN GSS")
DATA_DIR  = os.path.join(OUT, "02_Data")
CHART_DIR = os.path.join(OUT, "03_Grafik")

# Palet konsisten dengan make_deliverable.py
C_GREEN, C_SOCIAL, C_SUSTAIN, C_SL = "#2E7D32", "#C2A100", "#7B1FA2", "#1565C0"
C_NEUTRAL, C_ACCENT = "#9E9E9E", "#0056B2"

CLASS_COLOR = {
    "green": C_GREEN, "social": C_SOCIAL,
    "sustainability": C_SUSTAIN, "sustainability_linked": C_SL,
}
CLASS_LABEL = {
    "green": "Green", "social": "Social",
    "sustainability": "Sustainability", "sustainability_linked": "Sustainability-Linked",
}
CLASS_ORDER = ["green", "social", "sustainability", "sustainability_linked"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130,
})

TRILIUN = 1e12


# ---------------------------------------------------------------------------
def load_gss() -> list[dict]:
    rows = list(csv.DictReader(open(os.path.join(DATA, "universe_title_census.csv"),
                                    encoding="utf-8-sig")))
    gss = [r for r in rows if r["is_gss_titled"].strip().lower() in ("true", "1", "yes")]
    for r in gss:
        r["_class"] = r["title_gss_type"].strip().lower()
        r["_os"] = float(r["Outstanding"]) if r["Outstanding"] else 0.0
        m = re.search(r"Tahun\s+(20\d\d)", r["BondName"])
        r["_issue_year"] = int(m.group(1)) if m else None
        r["_mature_year"] = int(r["MatureDate"][:4]) if r["MatureDate"] else None
    return gss


# ---------------------------------------------------------------------------
def chart_issuance(gss: list[dict]) -> list[dict]:
    """Grafik 06 — penerbitan GSS per tahun: jumlah (stacked per kelas) + outstanding (garis)."""
    years = sorted({r["_issue_year"] for r in gss if r["_issue_year"]})
    n_by = {y: Counter() for y in years}
    os_by = defaultdict(float)
    for r in gss:
        y = r["_issue_year"]
        if y is None:
            continue
        n_by[y][r["_class"]] += 1
        os_by[y] += r["_os"]

    fig, ax1 = plt.subplots(figsize=(8.4, 5))
    bottom = [0] * len(years)
    for cls in CLASS_ORDER:
        vals = [n_by[y][cls] for y in years]
        ax1.bar([str(y) for y in years], vals, bottom=bottom,
                color=CLASS_COLOR[cls], label=CLASS_LABEL[cls],
                edgecolor="white", linewidth=0.6)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for i, y in enumerate(years):
        ax1.text(i, bottom[i] + 0.4, str(bottom[i]), ha="center",
                 fontsize=9, fontweight="bold")
    ax1.set_ylabel("Jumlah instrumen diterbitkan")
    ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylim(0, max(bottom) * 1.18)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    os_t = [os_by[y] / TRILIUN for y in years]
    ax2.plot([str(y) for y in years], os_t, color=C_ACCENT, marker="o",
             linewidth=2.2, label="Outstanding (Rp T)")
    ax2.set_ylabel("Outstanding diterbitkan (Rp Triliun)", color=C_ACCENT)
    ax2.tick_params(axis="y", labelcolor=C_ACCENT)
    ax2.set_ylim(0, max(os_t) * 1.25)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5,
               loc="upper left", ncol=2)
    ax1.set_title("Tren Penerbitan EBUS GSS Korporasi per Tahun")
    fig.text(0.5, -0.02,
             "Tahun = penanda 'Tahun YYYY' pada nama instrumen di listing BEI. "
             "2026 = sebagian tahun (data per 18 Jun 2026).",
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout()
    _save(fig, "06_tren_penerbitan_gss.png")

    return [{"tahun_terbit": y, "n_total": sum(n_by[y].values()),
             **{f"n_{c}": n_by[y][c] for c in CLASS_ORDER},
             "outstanding_triliun": round(os_by[y] / TRILIUN, 3)} for y in years]


# ---------------------------------------------------------------------------
def chart_maturity(gss: list[dict]) -> list[dict]:
    """Grafik 07 — profil jatuh tempo: outstanding GSS jatuh tempo per tahun."""
    by_year = defaultdict(float)
    cnt_year = Counter()
    for r in gss:
        y = r["_mature_year"]
        if y is None:
            continue
        by_year[y] += r["_os"]
        cnt_year[y] += 1
    years = sorted(by_year)

    fig, ax = plt.subplots(figsize=(8.4, 5))
    os_t = [by_year[y] / TRILIUN for y in years]
    xlabels = [f"{y}\n({cnt_year[y]} instr.)" for y in years]
    bars = ax.bar(xlabels, os_t, color=C_ACCENT, edgecolor="white", linewidth=0.6)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in os_t], padding=2,
                 fontsize=8.5, fontweight="bold")
    ax.set_ylabel("Outstanding jatuh tempo (Rp Triliun)")
    ax.set_ylim(0, max(os_t) * 1.15)
    ax.set_title("Profil Jatuh Tempo Outstanding GSS Korporasi (Maturity Ladder)")
    fig.text(0.5, -0.02,
             "Konsentrasi jatuh tempo menandai kebutuhan refinancing GSS pada tahun bersangkutan.",
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout()
    _save(fig, "07_profil_jatuh_tempo.png")

    return [{"tahun_jatuh_tempo": y, "n_instrumen": cnt_year[y],
             "outstanding_triliun": round(by_year[y] / TRILIUN, 3)} for y in years]


# ---------------------------------------------------------------------------
def chart_concentration(gss: list[dict]) -> tuple[float, list[dict]]:
    """Grafik 08 — konsentrasi penerbit (top + HHI) & sebaran rating."""
    os_by_issuer = defaultdict(float)
    for r in gss:
        os_by_issuer[r["IssuerCode"]] += r["_os"]
    total_os = sum(os_by_issuer.values())
    shares = {k: v / total_os for k, v in os_by_issuer.items()}
    hhi = sum((s * 100) ** 2 for s in shares.values())  # HHI 0–10.000
    top = sorted(os_by_issuer.items(), key=lambda x: x[1], reverse=True)[:8]

    # rating: kelompokkan ke pita utama (buang sufiks (sy)/(cg)/idn dll.)
    def rating_band(raw: str) -> str:
        raw = (raw or "").strip()
        if not raw:
            return "Tanpa rating"
        m = re.match(r"id\s*([A-D]{1,3}[+-]?)", raw)
        return m.group(1) if m else raw
    rat = Counter(rating_band(r["Rating"]) for r in gss)
    _ord = {"AAA": 0, "AA+": 1, "AA": 2, "AA-": 3, "A+": 4, "A": 5, "A-": 6}
    rat_items = sorted(rat.items(), key=lambda x: _ord.get(x[0], 99))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5),
                                   gridspec_kw={"width_ratios": [1.4, 1]})

    names = [k for k, _ in top][::-1]
    vals = [v / TRILIUN for _, v in top][::-1]
    bars = ax1.barh(names, vals, color=C_ACCENT)
    ax1.bar_label(bars, labels=[f"{v:.1f}" for v in vals], padding=3,
                  fontsize=8.5, fontweight="bold")
    top5_share = sum(s for _, s in sorted(shares.items(),
                     key=lambda x: x[1], reverse=True)[:5]) * 100
    ax1.set_title(f"Konsentrasi Penerbit (top 8 outstanding)\n"
                  f"HHI = {hhi:,.0f} · top-5 = {top5_share:.0f}% pasar GSS")
    ax1.set_xlabel("Outstanding (Rp Triliun)")
    ax1.tick_params(axis="y", labelsize=9)

    rnames = [k for k, _ in rat_items]
    rvals = [v for _, v in rat_items]
    rcol = [C_GREEN if n == "AAA" else C_ACCENT if n.startswith("AA")
            else C_SOCIAL for n in rnames]
    rb = ax2.bar(rnames, rvals, color=rcol, edgecolor="white", linewidth=0.6)
    ax2.bar_label(rb, padding=3, fontweight="bold")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title("Sebaran Rating Instrumen GSS")
    ax2.set_ylabel("Jumlah instrumen")
    ax2.tick_params(axis="x", labelsize=9)

    fig.suptitle("Struktur Pasar GSS Korporasi: Konsentrasi & Kualitas Kredit",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.01,
             "HHI > 2.500 = pasar sangat terkonsentrasi (acuan lazim regulator).",
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "08_konsentrasi_dan_rating.png")

    detail = ([{"metrik": "HHI_outstanding", "nilai": round(hhi, 1)},
               {"metrik": "top5_share_pct", "nilai": round(top5_share, 1)},
               {"metrik": "n_penerbit", "nilai": len(os_by_issuer)}]
              + [{"metrik": f"rating_{k}", "nilai": v} for k, v in rat_items])
    return hhi, detail


# ---------------------------------------------------------------------------
def _save(fig, name: str):
    path = os.path.join(CHART_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  grafik  -> {os.path.relpath(path, ROOT)}")


def write_csv(issuance, maturity, conc_detail):
    path = os.path.join(DATA_DIR, "06_statistik_deskriptif_gss.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["# PENERBITAN PER TAHUN"])
        w.writerow(["tahun_terbit", "n_total", *[f"n_{c}" for c in CLASS_ORDER],
                    "outstanding_triliun"])
        for r in issuance:
            w.writerow([r["tahun_terbit"], r["n_total"],
                        *[r[f"n_{c}"] for c in CLASS_ORDER], r["outstanding_triliun"]])
        w.writerow([])
        w.writerow(["# PROFIL JATUH TEMPO"])
        w.writerow(["tahun_jatuh_tempo", "n_instrumen", "outstanding_triliun"])
        for r in maturity:
            w.writerow([r["tahun_jatuh_tempo"], r["n_instrumen"], r["outstanding_triliun"]])
        w.writerow([])
        w.writerow(["# KONSENTRASI & RATING"])
        w.writerow(["metrik", "nilai"])
        for r in conc_detail:
            w.writerow([r["metrik"], r["nilai"]])
    print(f"  csv     -> {os.path.relpath(path, ROOT)}")


def _self_check(gss):
    # Tahun penerbitan harus terbaca untuk SELURUH 82 — fondasi grafik tren.
    miss = [r["BondName"] for r in gss if r["_issue_year"] is None]
    assert not miss, f"{len(miss)} instrumen tanpa 'Tahun YYYY' di nama: {miss[:3]}"
    # Total harus konsisten dengan sensus (82 GSS).
    assert len(gss) == 82, f"GSS-titled = {len(gss)}, diharapkan 82 (sensus berubah?)"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)
    print("=== STATISTIK DESKRIPTIF PASAR GSS ===\n")
    gss = load_gss()
    _self_check(gss)
    print(f"[1/4] {len(gss)} instrumen GSS dimuat (dari universe).")
    issuance = chart_issuance(gss)
    maturity = chart_maturity(gss)
    hhi, conc_detail = chart_concentration(gss)
    write_csv(issuance, maturity, conc_detail)
    print(f"\n[selesai] HHI={hhi:,.0f}  "
          f"penerbitan {issuance[0]['tahun_terbit']}–{issuance[-1]['tahun_terbit']}, "
          f"puncak {max(issuance, key=lambda r: r['n_total'])['tahun_terbit']}")


if __name__ == "__main__":
    main()
