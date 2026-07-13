"""
OCR PDF hasil scan di korpus mentah, cocokkan ke instrumen GSS listing IDX,
dan salin (PDF + sidecar .txt hasil OCR) ke gold corpus.

Pelengkap curate_missing_gss.py untuk pengumuman yang hanya tersedia sebagai
gambar (Bukti Iklan, sebagian Informasi Tambahan). OCR lokal via RapidOCR
(onnxruntime) — gratis, offline.

Jalankan dari root repo:
    python code/idx_prospektus_scraper/ocr_scanned_gss.py --codes BJBR BMRI ...
"""
from __future__ import annotations
import argparse, os, re, shutil, sys

ROOT = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
sys.path.insert(0, os.path.join(ROOT, "code"))

from evaluation.build_gold_db import _match_idx_all, _build_idx_lookup

RAW_DIR  = os.path.join(ROOT, "code", "pdf_by_content", "01_prospektus_utama")
GOLD_GSS = os.path.join(ROOT, "data", "pdf_by_content", "01_prospektus_utama",
                        "0. Fix GSS", "GSS")
LP = "\\\\?\\"
_FOLDER_RE = re.compile(r"^(\d{8})_([A-Z]{4})_")

_ocr = None


def ocr_cover(pdf_path: str, n_pages: int = 3, dpi: int = 150) -> str:
    """OCR halaman-halaman awal sebuah PDF scan."""
    global _ocr
    import fitz
    import numpy as np
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    out: list[str] = []
    for p in (LP + pdf_path, pdf_path):
        try:
            doc = fitz.open(p)
        except Exception:
            continue
        try:
            for i in range(min(n_pages, len(doc))):
                pix = doc[i].get_pixmap(dpi=dpi)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = img[:, :, :3]
                res, _ = _ocr(img)
                if res:
                    out.append("\n".join(seg[1] for seg in res))
        finally:
            doc.close()
        break
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    codes = {c.upper() for c in args.codes}
    idx_lookup = _build_idx_lookup()

    import fitz
    copied = no_match = 0
    for folder in sorted(os.listdir(RAW_DIR)):
        m = _FOLDER_RE.match(folder)
        if not m or m.group(2) not in codes:
            continue
        date_str, issuer = m.groups()
        fpath = os.path.join(RAW_DIR, folder)
        if not os.path.isdir(fpath):
            continue
        for pdf in sorted(os.listdir(LP + fpath)):
            if not pdf.lower().endswith(".pdf"):
                continue
            src = os.path.join(fpath, pdf)
            # hanya PDF yang TIDAK punya teks (scan)
            try:
                doc = fitz.open(LP + src)
                has_text = any(doc[i].get_text().strip()
                               for i in range(min(3, len(doc))))
                doc.close()
            except Exception:
                continue
            if has_text:
                continue
            dst_name = pdf if _FOLDER_RE.match(pdf) else f"{date_str}_{issuer}_{pdf}"
            dst = os.path.join(GOLD_GSS, issuer, dst_name)
            if os.path.exists(LP + dst):
                continue
            text = ocr_cover(src)
            if not text.strip():
                continue
            matched = _match_idx_all(issuer, text, idx_lookup)
            if not matched:
                no_match += 1
                continue
            print(f"[{issuer}] {dst_name[:75]}")
            for t in matched[:3]:
                print(f"   -> {t['BondName'][:70]}")
            if not args.dry_run:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(LP + src, LP + dst)
                with open(LP + dst + ".txt", "w", encoding="utf-8") as f:
                    f.write(text)
            copied += 1

    print(f"\n{'DRY-RUN: ' if args.dry_run else ''}{copied} PDF scan ter-OCR & "
          f"tersalin | {no_match} scan tanpa match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
