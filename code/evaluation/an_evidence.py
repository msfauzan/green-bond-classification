"""
Pembangun bukti empiris untuk Analytical Note §8-§9.

Menghasilkan dua CSV:
  data/market_census.csv         -- sensus pasar 82/1.437 per kelas GSS
  data/sector_decomposition.csv  -- dekomposisi sektoral 27 gold GSS bond

Jalankan dari root repo:
  python evaluation/an_evidence.py
"""
from __future__ import annotations
import csv
import os
import sys
from collections import defaultdict

ROOT     = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
CODE_DIR = os.path.join(ROOT, "code")
sys.path.insert(0, CODE_DIR)

from classifier.engine       import read_pdf_text
from classifier.ml_engine    import classify_ml
from classifier.taxonomy     import ALL_CATEGORIES, GSSClass
from classifier.title_lookup import gss_type_from_title

DATA_DIR  = os.path.join(ROOT, "data")
BASE_PROS = os.path.join(DATA_DIR, "pdf_by_content", "01_prospektus_utama")
GOLD_GSS  = os.path.join(BASE_PROS, "0. Fix GSS", "GSS")
MAX_PAGES = 50
LP = "\\\\.?\\"          # prefiks long-path Windows (\\?\)


# ---------------------------------------------------------------------------
# Long-path helpers (identik dgn compare_engines.py)
# ---------------------------------------------------------------------------
def _ll(path):
    for p in (path, LP + path):
        try:
            return os.listdir(p)
        except OSError:
            continue
    return []

def _isdir(p):
    return os.path.isdir(p) or os.path.isdir(LP + p)

def _sz(p):
    for q in (p, LP + p):
        try:
            return os.path.getsize(q)
        except OSError:
            continue
    return 0

def _all_pdfs(root):
    out = []
    for e in _ll(root):
        full = os.path.join(root, e)
        if _isdir(full):
            out.extend(_all_pdfs(full))
        elif e.lower().endswith(".pdf"):
            out.append(full)
    return out


# ---------------------------------------------------------------------------
# (1) Sensus pasar — dari listing IDX yang sudah ada (tanpa ML)
# ---------------------------------------------------------------------------
def market_census() -> list[dict]:
    from classifier.title_lookup import IDX_CSV as universe_path
    gss_path = os.path.join(DATA_DIR, "idx_gss_all_20260618_140427.csv")

    n_universe = 0
    with open(universe_path, encoding="utf-8-sig") as f:
        for _ in csv.DictReader(f):
            n_universe += 1

    # Tipe GSS diambil dari NAMA instrumen (otoritatif), bukan kolom gss_type cached
    # di idx_gss_all yang memetakan SLCN (Sustainability UoP) ke "Sustainability Linked".
    # gss_type_from_title: "terkait keberlanjutan" → sustainability_linked;
    #                      "berlandaskan/obligasi keberlanjutan" → sustainability.
    DISPLAY = {
        "green":                "Green",
        "social":               "Social",
        "sustainability":       "Sustainability",
        "sustainability_linked":"Sustainability Linked",
    }
    ORDER = ["green", "social", "sustainability", "sustainability_linked"]

    type_count: dict[str, int] = defaultdict(int)
    type_outstanding: dict[str, float] = defaultdict(float)

    with open(gss_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            gt = gss_type_from_title(row["BondName"]) or "unknown"
            type_count[gt] += 1
            try:
                type_outstanding[gt] += float(row["Outstanding"] or 0)
            except ValueError:
                pass

    n_gss = sum(type_count.values())
    total_os = sum(type_outstanding.values())

    out_rows = []
    for gt in ORDER:
        if not type_count[gt]:
            continue
        cnt    = type_count[gt]
        os_idr = type_outstanding[gt]
        out_rows.append({
            "gss_type"                : DISPLAY[gt],
            "n_instrumen"             : cnt,
            "total_outstanding_triliun": round(os_idr / 1e12, 2),
            "share_universe_pct"      : round(cnt / n_universe * 100, 2),
        })
    out_rows.append({
        "gss_type"                : "TOTAL",
        "n_instrumen"             : n_gss,
        "total_outstanding_triliun": round(total_os / 1e12, 2),
        "share_universe_pct"      : round(n_gss / n_universe * 100, 2),
        "n_universe"              : n_universe,
    })

    out_path = os.path.join(DATA_DIR, "market_census.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        # Tulis fieldnames dari baris pertama; TOTAL row punya kolom ekstra n_universe
        fields = list(out_rows[0].keys()) + ["n_universe"]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)

    print(f"\n=== SENSUS PASAR GSS KORPORASI (TIPE DARI NAMA) ===")
    print(f"  Universe EBUS korporasi : {n_universe}")
    print(f"  GSS berlabel (IDX)      : {n_gss}  ({n_gss / n_universe * 100:.2f}% universe)")
    print(f"  Breakdown per kelas:")
    for row in out_rows[:-1]:
        print(
            f"    {row['gss_type']:<24}: {row['n_instrumen']:>3} instrumen, "
            f"Rp {row['total_outstanding_triliun']:>8.2f} T  "
            f"({row['share_universe_pct']:.2f}% universe)"
        )
    print(f"  Total outstanding GSS   : Rp {round(total_os / 1e12, 2):.2f} T")
    print(f"  => {out_path}")
    return out_rows


# ---------------------------------------------------------------------------
# (2) Dekomposisi sektoral — ML engine pada 27 gold GSS PDF
# ---------------------------------------------------------------------------
def _build_gss_gold() -> list[dict]:
    rows = []
    for iss in sorted(_ll(GOLD_GSS)):
        ip = os.path.join(GOLD_GSS, iss)
        if not _isdir(ip):
            continue
        for pdf in _all_pdfs(ip):
            rows.append({"issuer": iss, "path": pdf})
    return rows


def sector_decomposition() -> tuple[list[dict], dict, dict]:
    gold = _build_gss_gold()
    cat_label = {c.key: c.name_id for c in ALL_CATEGORIES}

    detail_rows: list[dict] = []
    sector_counts: dict[str, int] = {c.key: 0 for c in ALL_CATEGORIES}
    bucket_counts = {"terdekomposisi": 0, "sektor_tak_terverifikasi": 0, "level0": 0}

    for i, r in enumerate(gold, 1):
        print(f"  [{i:2}/{len(gold)}] {r['issuer']:6}  {os.path.basename(r['path'])[:55]}")
        text = read_pdf_text(r["path"], MAX_PAGES)
        rm = classify_ml(text, issuer=r["issuer"])
        if rm is None:
            print("         SKIP (teks kosong)")
            continue

        # Tentukan ember (prioritas: Level-0 > fallback judul > semantic)
        if rm.level0_evidence:
            bucket = "level0"
        elif rm.needs_review or rm.sector_unspecified:
            bucket = "sektor_tak_terverifikasi"
        else:
            bucket = "terdekomposisi"

        bucket_counts[bucket] += 1
        sector_keys = rm.sector_keys()

        if bucket == "terdekomposisi":
            for k in sector_keys:
                if k in sector_counts:
                    sector_counts[k] += 1

        # Verifikasi klaim: konsistensi antara nama/jenis instrumen vs bukti UoP
        if bucket == "level0":
            vs = "Level-0 (struktural)"
        elif bucket == "sektor_tak_terverifikasi":
            vs = "Tidak terverifikasi (UoP tipis)"
        else:
            has_env = bool(rm.decomp_env)
            has_soc = bool(rm.decomp_soc)
            if rm.gss_class == GSSClass.GREEN:
                vs = "Terverifikasi" if has_env else ("Tidak konsisten" if has_soc else "Tidak tersubstansiasi")
            elif rm.gss_class == GSSClass.SOCIAL:
                vs = "Terverifikasi" if has_soc else ("Tidak konsisten" if has_env else "Tidak tersubstansiasi")
            elif rm.gss_class == GSSClass.SUSTAINABILITY:
                vs = "Terverifikasi" if (has_env and has_soc) else ("Sebagian" if (has_env or has_soc) else "Tidak tersubstansiasi")
            else:
                vs = "Terverifikasi"

        detail_rows.append({
            "issuer"              : r["issuer"],
            "pdf"                 : os.path.basename(r["path"]),
            "gss_class"           : rm.gss_class.value,
            "bucket"              : bucket,
            "sector_keys"         : "|".join(sector_keys),
            "needs_review"        : rm.needs_review,
            "sector_unspecified"  : rm.sector_unspecified,
            "level0_evidence"     : "|".join(rm.level0_evidence[:3]),
            "confidence"          : rm.confidence,
            "verification_status" : vs,
        })

    out_path = os.path.join(DATA_DIR, "sector_decomposition.csv")
    if detail_rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader()
            w.writerows(detail_rows)

    n = len(detail_rows)
    print(f"\n=== DEKOMPOSISI SEKTORAL ({n} gold GSS) ===")
    print(f"  Ember:")
    for bkt, cnt in bucket_counts.items():
        pct = cnt / n * 100 if n else 0
        print(f"    {bkt:<30}: {cnt:>2} ({pct:.0f}%)")

    active = {k: v for k, v in sector_counts.items() if v > 0}
    if active:
        print(f"\n  Kategori (bond 'terdekomposisi' = {bucket_counts['terdekomposisi']}):")
        for key, cnt in sorted(active.items(), key=lambda x: -x[1]):
            print(f"    {cat_label[key]:<45} ({key}): {cnt}")

    print(f"  => {out_path}")
    return detail_rows, sector_counts, bucket_counts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PEMBANGUNAN BUKTI AN §8-§9 ===\n")

    print("(1) Sensus pasar (dari listing IDX — tanpa ML)...")
    market_census()

    print("\n(2) Dekomposisi sektoral (ML engine pada gold GSS PDF)...")
    print("    Model ~120 MB diunduh otomatis pada pertama kali.\n")
    sector_decomposition()

    print("\nSelesai.")
