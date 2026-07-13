"""
Kurasi otomatis: salin PDF prospektus hasil scrape yang sampulnya GSS
ke gold corpus 0. Fix GSS/GSS/<ISSUER>/.

Gate (konservatif, anti-jebakan PUB "Berkelanjutan"):
  sampul (1500 char pertama) harus mengandung TITLE_GSS_MARKERS
  (via gss_title_in_text — "berkelanjutan" polos TIDAK termasuk marker).
Pemetaan PDF -> seri instrumen dilakukan build_gold_db.py, bukan di sini;
penempatan cukup per emiten.

Jalankan dari root repo:
    python code/idx_prospektus_scraper/curate_missing_gss.py [--dry-run]
"""
from __future__ import annotations
import argparse, os, re, shutil, sys

ROOT = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
sys.path.insert(0, os.path.join(ROOT, "code"))

from classifier.title_lookup import all_instruments, gss_title_in_text
from evaluation.build_gold_db import (
    _read_cover, _match_idx_all, _build_idx_lookup, doc_type, _n_pages,
)

RAW_DIR  = os.path.join(ROOT, "code", "pdf_by_content", "01_prospektus_utama")
GOLD_GSS = os.path.join(ROOT, "data", "pdf_by_content", "01_prospektus_utama",
                        "0. Fix GSS", "GSS")
LP = "\\\\?\\"

_FOLDER_RE = re.compile(r"^(\d{8})_([A-Z]{4})_")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--codes", nargs="+", default=None,
                    help="Batasi ke kode emiten tertentu")
    args = ap.parse_args()
    codes = {c.upper() for c in args.codes} if args.codes else None
    idx_lookup = _build_idx_lookup()

    copied = skipped_nongss = skipped_dup = unreadable = 0
    for folder in sorted(os.listdir(RAW_DIR)):
        m = _FOLDER_RE.match(folder)
        if not m:
            continue
        date_str, issuer = m.groups()
        if codes and issuer not in codes:
            continue
        fpath = os.path.join(RAW_DIR, folder)
        if not os.path.isdir(fpath):
            continue
        for pdf in sorted(os.listdir(LP + fpath)):
            if not pdf.lower().endswith(".pdf"):
                continue
            src = os.path.join(fpath, pdf)
            text = _read_cover(src)
            if not text.strip():
                unreadable += 1
                print(f"[{issuer}] TAK TERBACA: {pdf[:70]}")
                continue
            # Gate = matcher gold DB: PDF hanya masuk bila sampulnya cocok
            # dengan minimal satu instrumen GSS di listing IDX (issuer,
            # marker, tahun, tahap, seri, jenis instrumen).
            matched = _match_idx_all(issuer, text, idx_lookup)
            if not matched:
                skipped_nongss += 1
                continue
            dtype, rank = doc_type(text, _n_pages(src))
            if rank < 0:   # pemeringkatan dkk yang menyebut nama obligasi
                skipped_nongss += 1
                continue
            markers = [m["BondName"][:40] for m in matched[:2]]
            # nama file scraper sudah berformat YYYYMMDD_CODE_...; jangan dobel
            dst_name = pdf if _FOLDER_RE.match(pdf) else f"{date_str}_{issuer}_{pdf}"
            dst_dir = os.path.join(GOLD_GSS, issuer)
            dst = os.path.join(dst_dir, dst_name)
            if os.path.exists(LP + dst):
                skipped_dup += 1
                continue
            print(f"[{issuer}] {dst_name[:80]}")
            print(f"   marker: {', '.join(markers[:4])}")
            if not args.dry_run:
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(LP + src, LP + dst)
            copied += 1

    print(f"\n{'DRY-RUN: ' if args.dry_run else ''}{copied} PDF GSS tersalin | "
          f"{skipped_nongss} non-GSS dilewati | {skipped_dup} sudah ada | "
          f"{unreadable} tak terbaca (image PDF?)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
