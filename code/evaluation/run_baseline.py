"""
Baseline rule-based: ukur engine taxonomy-grounded terhadap gold corpus.

Gold set (lihat keputusan 2026-06-20):
  - POSITIF (GSS) : 12 emiten / prospektus terverifikasi tangan di
                    pdf_by_content/01_prospektus_utama/0. Fix GSS/GSS/
  - NEGATIF (NonGSS): di-SAMPEL dari korpus prospektus yang sudah diunduh
                    (01_prospektus_utama), 1 prospektus per emiten non-GSS.
                    *Asumsi konvensional di tingkat emiten — bukan verifikasi
                    per-dokumen.* Cukup untuk mengukur precision/false-positive
                    (termasuk jebakan "Berkelanjutan").

Output:
  - ringkasan confusion matrix + precision/recall (biner GSS vs NonGSS)
  - distribusi sub-kelas pada positif (Green/Social/Sustainability/SL/Wakaf)
  - daftar kasus salah (FP/FN) untuk analisis kesalahan
  - data/baseline_results.csv (detail per dokumen)
"""
from __future__ import annotations
import os
import sys
import csv

ROOT = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
sys.path.insert(0, ROOT)

from classifier.engine import read_pdf_text, classify, LP  # noqa: E402
from classifier.taxonomy import GSSClass  # noqa: E402

BASE_PROS = os.path.join(ROOT, "pdf_by_content", "01_prospektus_utama")
GOLD_GSS = os.path.join(BASE_PROS, "0. Fix GSS", "GSS")
GSS_ISSUERS = {"ARKO", "BBNI", "BBRI", "BBTN", "BJBR", "BMRI",
               "BRIS", "OPPM", "PNMP", "POLI", "PPGD", "SMII"}
MAX_PAGES = 50


# --- helper long-path ---------------------------------------------------------
def _ll(path: str) -> list[str]:
    for p in (path, LP + path):
        try:
            return os.listdir(p)
        except OSError:
            continue
    return []


def _isdir(path: str) -> bool:
    return os.path.isdir(path) or os.path.isdir(LP + path)


def _size(path: str) -> int:
    for p in (path, LP + path):
        try:
            return os.path.getsize(p)
        except OSError:
            continue
    return 0


def _largest_pdf(folder: str) -> str | None:
    pdfs = [(_size(os.path.join(folder, f)), f) for f in _ll(folder) if f.lower().endswith(".pdf")]
    pdfs.sort(reverse=True)
    return os.path.join(folder, pdfs[0][1]) if pdfs else None


def _all_pdfs(root: str) -> list[str]:
    """Semua PDF di bawah root (rekursif, long-path safe)."""
    out: list[str] = []
    for e in _ll(root):
        full = os.path.join(root, e)
        if _isdir(full):
            out.extend(_all_pdfs(full))
        elif e.lower().endswith(".pdf"):
            out.append(full)
    return out


def _main_prospektus(issuer_dir: str) -> list[str]:
    """Kembalikan PDF utama per pengumuman (nested) atau PDF terbesar (flat)."""
    entries = _ll(issuer_dir)
    subdirs = [e for e in entries if _isdir(os.path.join(issuer_dir, e))]
    out: list[str] = []
    if subdirs:
        for sd in sorted(subdirs):
            p = _largest_pdf(os.path.join(issuer_dir, sd))
            if p:
                out.append(p)
    else:
        p = _largest_pdf(issuer_dir)
        if p:
            out.append(p)
    return out


# --- bangun gold set ----------------------------------------------------------
def build_gold() -> list[dict]:
    rows: list[dict] = []
    # positif
    for iss in sorted(_ll(GOLD_GSS)):
        ip = os.path.join(GOLD_GSS, iss)
        if not _isdir(ip):
            continue
        for pdf in _all_pdfs(ip):           # semua dok GSS terkurasi (27 file)
            rows.append({"issuer": iss, "gold": "GSS", "source": "gold", "path": pdf})

    # negatif: 1 prospektus per emiten non-GSS dari korpus luas
    by_issuer: dict[str, list[str]] = {}
    for name in sorted(_ll(BASE_PROS)):
        full = os.path.join(BASE_PROS, name)
        if name == "0. Fix GSS" or not _isdir(full):
            continue
        parts = name.split("_")
        iss = parts[1] if len(parts) >= 2 else "?"
        if iss in GSS_ISSUERS:
            continue
        by_issuer.setdefault(iss, []).append(full)
    for iss, folders in sorted(by_issuer.items()):
        # pilih folder dgn PDF terbesar (kemungkinan prospektus utama, bukan iklan)
        best = max(folders, key=lambda f: _size(_largest_pdf(f) or ""))
        pdf = _largest_pdf(best)
        if pdf:
            rows.append({"issuer": iss, "gold": "NonGSS", "source": "sampled", "path": pdf})
    return rows


# --- jalankan & evaluasi ------------------------------------------------------
def main() -> None:
    gold = build_gold()
    n_pos = sum(1 for r in gold if r["gold"] == "GSS")
    n_neg = sum(1 for r in gold if r["gold"] == "NonGSS")
    print(f"Gold set: {len(gold)} dokumen  ({n_pos} GSS positif, {n_neg} NonGSS sampel)")
    print("Memproses PDF...\n")

    tp = fp = fn = tn = 0
    skipped: list[dict] = []
    fp_cases: list[dict] = []
    fn_cases: list[dict] = []
    subclass = {}
    out_rows: list[dict] = []

    for r in gold:
        text = read_pdf_text(r["path"], MAX_PAGES)
        res = classify(text)
        if res is None:
            skipped.append(r)
            continue
        pred = "GSS" if res.is_gss else "NonGSS"
        if r["gold"] == "GSS":
            subclass[res.gss_class.value] = subclass.get(res.gss_class.value, 0) + 1
            if pred == "GSS":
                tp += 1
            else:
                fn += 1
                fn_cases.append({**r, "res": res})
        else:
            if pred == "GSS":
                fp += 1
                fp_cases.append({**r, "res": res})
            else:
                tn += 1

        out_rows.append({
            "issuer": r["issuer"], "gold": r["gold"], "source": r["source"],
            "pred_class": res.gss_class.value, "pred_binary": pred,
            "correct": (pred == r["gold"]),
            "sectors": "|".join(res.sector_keys()),
            "blue": "|".join(res.blue),
            "level0": "|".join(res.level0_evidence),
            "anchor": res.anchor, "confidence": res.confidence,
            "pdf": os.path.basename(r["path"]),
        })

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    print("=" * 64)
    print("  BASELINE RULE-BASED — biner GSS vs NonGSS")
    print("=" * 64)
    print(f"  dievaluasi : {tp+fp+fn+tn}   (dilewati/PDF gambar: {len(skipped)})")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  Precision  : {prec:.2f}")
    print(f"  Recall     : {rec:.2f}")
    print(f"  F1         : {f1:.2f}")
    print()
    print("  Distribusi sub-kelas pada GSS positif:")
    for k, v in sorted(subclass.items(), key=lambda x: -x[1]):
        print(f"     {k:24} {v}")

    if fn_cases:
        print("\n  FALSE NEGATIVE (GSS gold -> diprediksi NonGSS):")
        for c in fn_cases:
            print(f"     {c['issuer']:6} anchor='{c['res'].anchor[:30]}'  {os.path.basename(c['path'])[:50]}")
    if fp_cases:
        print("\n  FALSE POSITIVE (NonGSS -> diprediksi GSS):")
        for c in fp_cases:
            sect = "|".join(c["res"].sector_keys()) or "|".join(c["res"].level0_evidence)
            print(f"     {c['issuer']:6} -> {c['res'].gss_class.value:16} [{sect[:40]}]  {os.path.basename(c['path'])[:40]}")
    if skipped:
        print("\n  DILEWATI (teks kosong / perlu OCR):")
        for c in skipped:
            print(f"     {c['issuer']:6} {os.path.basename(c['path'])[:55]}")

    out_csv = os.path.join(ROOT, "data", "baseline_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n  Detail per dokumen -> {out_csv}")


if __name__ == "__main__":
    main()
