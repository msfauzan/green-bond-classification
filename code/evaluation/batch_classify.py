"""
Klasifikasi ML secara batch atas prospektus yang sudah di-scrape.

Struktur korpus yang dikenali:
  data/pdf_by_content/01_prospektus_utama/YYYYMMDD_EMITEN_Judul/*.pdf
  code/pdf_by_content/01_prospektus_utama/YYYYMMDD_EMITEN_Judul/*.pdf

Emiten diekstrak dari nama folder: bagian ke-2 setelah split '_'
  "20240103_WIKA_Penyampaian Prospektus" → issuer = "WIKA"

Cara pakai:
  python code/evaluation/batch_classify.py

Fitur:
  - Checkpoint: lewati PDF yang sudah ada di CSV output (aman dijalankan ulang)
  - Lewati emiten gold set (sudah ada di comparison_results.csv)
  - Lewati subfolder "0. Fix GSS" / "0. GSS" (gold corpus)
  - Prefix \\\\?\\ untuk semua path (Windows 260-char limit)

Output:
  data/batch_classify_results.csv  (append / checkpoint-aware)
"""
from __future__ import annotations
import csv
import os
import re
import sys
from collections import defaultdict

ROOT     = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
CODE_DIR = os.path.join(ROOT, "code")
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, CODE_DIR)

from classifier.engine       import read_pdf_text          # noqa: E402
from classifier.ml_engine    import classify_ml, MLResult  # noqa: E402
from classifier.taxonomy     import GSSClass               # noqa: E402

LP = "\\\\?\\"   # Windows long-path prefix

# Lokasi korpus prospektus (urutan: scan semua)
CORPUS_ROOTS: list[str] = [
    os.path.join(DATA_DIR, "pdf_by_content", "01_prospektus_utama"),
    os.path.join(CODE_DIR, "pdf_by_content", "01_prospektus_utama"),
]

# Subfolder yang berisi gold corpus — jangan di-scan
SKIP_DIR_PREFIXES = ("0.",)

# Emiten gold set — sudah ada di comparison_results.csv, lewati
GOLD_ISSUERS = {
    "ARKO","BBNI","BBRI","BBTN","BJBR","BMRI",
    "BRIS","OPPM","PNMP","POLI","PPGD","SMII",
    "SMFP","IIFF","ISSP",
}

MAX_PAGES = 50
OUT_CSV   = os.path.join(DATA_DIR, "batch_classify_results.csv")

FIELDNAMES = [
    "issuer","folder","pdf_filename",
    "title_gss","ml_pred","ml_class","confidence",
    "top_env","top_soc","needs_review","sector_keys",
    "framing_body","framing_title","pdf_full_path",
]

# Pola tanggal di awal nama folder: YYYYMMDD_
_DATE_PREFIX = re.compile(r"^\d{8}_")


# ---------------------------------------------------------------------------
# Utilitas path
# ---------------------------------------------------------------------------

def _ll(path: str) -> list[str]:
    for p in (path, LP + path):
        try:
            return os.listdir(p)
        except OSError:
            continue
    return []


def _issuer_from_folder(folder_name: str) -> str | None:
    """Ekstrak kode emiten dari nama folder YYYYMMDD_EMITEN_Judul."""
    if not _DATE_PREFIX.match(folder_name):
        return None
    parts = folder_name.split("_")
    if len(parts) < 2:
        return None
    return parts[1].upper().strip()


def _should_skip_dir(name: str) -> bool:
    return any(name.startswith(p) for p in SKIP_DIR_PREFIXES)


def _pdfs_in_folder(folder_path: str) -> list[str]:
    """Kembalikan daftar path PDF langsung di dalam folder (1 level)."""
    pdfs = []
    for fname in _ll(folder_path):
        if fname.lower().endswith(".pdf"):
            pdfs.append(os.path.join(folder_path, fname))
    return pdfs


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def _load_checkpoint() -> set[str]:
    done: set[str] = set()
    if not os.path.exists(OUT_CSV):
        return done
    with open(OUT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            p = row.get("pdf_full_path", "")
            if p:
                done.add(p)
    return done


# ---------------------------------------------------------------------------
# Klasifikasi satu PDF
# ---------------------------------------------------------------------------

def _classify_one(pdf_path: str, issuer: str, folder: str) -> dict | None:
    text = read_pdf_text(pdf_path, max_pages=MAX_PAGES)
    if not text.strip():
        return None

    result: MLResult | None = classify_ml(text, issuer=issuer)
    if result is None:
        return None

    top_env_str = "|".join(f"{k}:{s:.3f}" for k, s in result.top_env[:3])
    top_soc_str = "|".join(f"{k}:{s:.3f}" for k, s in result.top_soc[:3])

    title_gss_label = (
        "True"  if result.title_gss is True  else
        "False" if result.title_gss is False else
        "None"
    )

    return {
        "issuer":        issuer,
        "folder":        folder,
        "pdf_filename":  os.path.basename(pdf_path)[:80],
        "title_gss":     title_gss_label,
        "ml_pred":       "GSS" if result.is_gss else "NonGSS",
        "ml_class":      result.gss_class.value,
        "confidence":    f"{result.confidence:.4f}",
        "top_env":       top_env_str,
        "top_soc":       top_soc_str,
        "needs_review":  "True" if result.needs_review else "False",
        "sector_keys":   "|".join(result.sector_keys()[:5]),
        "framing_body":  "|".join(result.framing_body[:3]),
        "framing_title": "|".join(result.framing_title[:3]),
        "pdf_full_path": pdf_path,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    checkpoint = _load_checkpoint()
    print(f"Checkpoint: {len(checkpoint)} PDF sudah diproses sebelumnya")

    # Buka file output (append mode)
    write_header = not os.path.exists(OUT_CSV)
    out_f = open(OUT_CSV, "a", newline="", encoding="utf-8-sig")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    total_processed = total_gss = total_skip = total_err = 0
    issuer_gss: dict[str, list[str]] = defaultdict(list)

    for corpus_root in CORPUS_ROOTS:
        if not os.path.isdir(corpus_root):
            print(f"(lewati — tidak ada): {corpus_root}")
            continue
        print(f"\n=== Korpus: {os.path.relpath(corpus_root, ROOT)} ===")

        for folder_name in sorted(_ll(corpus_root)):
            # Lewati subfolder gold corpus
            if _should_skip_dir(folder_name):
                continue

            folder_path = os.path.join(corpus_root, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Ekstrak kode emiten
            issuer = _issuer_from_folder(folder_name)
            if issuer is None:
                continue
            if issuer in GOLD_ISSUERS:
                continue

            pdfs = _pdfs_in_folder(folder_path)
            if not pdfs:
                continue

            for pdf_path in pdfs:
                if pdf_path in checkpoint:
                    total_skip += 1
                    continue
                try:
                    row = _classify_one(pdf_path, issuer, folder_name)
                    if row is None:
                        total_err += 1
                        print(f"  KOSONG  {issuer} | {os.path.basename(pdf_path)[:55]}")
                        continue
                    writer.writerow(row)
                    out_f.flush()
                    checkpoint.add(pdf_path)
                    total_processed += 1
                    if row["ml_pred"] == "GSS":
                        total_gss += 1
                        issuer_gss[issuer].append(row["ml_class"])
                        print(f"  GSS [{row['ml_class']:15}] conf={row['confidence']}  "
                              f"{issuer} | {os.path.basename(pdf_path)[:45]}")
                except Exception as exc:
                    total_err += 1
                    print(f"  ERROR   {issuer} | {os.path.basename(pdf_path)[:45]}: {exc}")

    out_f.close()

    # Ringkasan
    print(f"\n{'='*55}")
    print(f"BATCH SELESAI")
    print(f"  Diproses   : {total_processed}")
    print(f"  Kandidat GSS: {total_gss}")
    print(f"  Skip (done) : {total_skip}")
    print(f"  Gagal baca  : {total_err}")
    if issuer_gss:
        print(f"\n  Emiten dengan kandidat GSS:")
        for iss, classes in sorted(issuer_gss.items()):
            print(f"    {iss:<10} {classes}")
    print(f"\n  Output -> {os.path.relpath(OUT_CSV, ROOT)}")
    print(f"  Selanjutnya: python code/evaluation/triage_candidates.py")


if __name__ == "__main__":
    main()
