"""
Saring kandidat GSS tidak berlabel dari hasil batch_classify.py.

Filter: title_gss IS NOT True (bukan instrumen berlabel) DAN ml_pred = GSS
        DAN confidence >= ambang (default 0.45).

Output:
  data/unlabeled_gss_candidates.csv  -- kandidat, urutkan confidence desc

Kolom tambahan untuk triase manual:
  manual_review_result   -- diisi tangan: confirmed / rejected / ambiguous
  manual_notes           -- catatan singkat dari reviewer

Jalankan setelah batch_classify.py selesai (minimal 1 sesi):
  python code/evaluation/triage_candidates.py
"""
from __future__ import annotations
import csv
import os
import sys
from collections import Counter, defaultdict

ROOT     = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
DATA_DIR = os.path.join(ROOT, "data")

IN_CSV  = os.path.join(DATA_DIR, "batch_classify_results.csv")
OUT_CSV = os.path.join(DATA_DIR, "unlabeled_gss_candidates.csv")

# Ambang confidence minimum untuk masuk daftar kandidat
DEFAULT_CONFIDENCE = 0.45

FIELDNAMES_OUT = [
    "issuer","bond_name_approx","pdf_full_path",
    "title_gss","ml_class","confidence",
    "top_env","top_soc","sector_keys",
    "framing_body","framing_title",
    "manual_review_result",   # diisi tangan: confirmed / rejected / ambiguous
    "manual_notes",
]


def main(min_confidence: float = DEFAULT_CONFIDENCE):
    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} tidak ditemukan. Jalankan batch_classify.py dulu.")
        sys.exit(1)

    with open(IN_CSV, encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))

    print(f"Total baris batch_classify_results.csv : {len(all_rows):,}")

    # Baca kandidat yang sudah punya hasil triase manual (bila ada)
    existing_triage: dict[str, dict] = {}
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                existing_triage[row.get("pdf_full_path", row.get("pdf_path", ""))] = {
                    "manual_review_result": row.get("manual_review_result",""),
                    "manual_notes":         row.get("manual_notes",""),
                }
        print(f"Triase manual sebelumnya terbaca : {len(existing_triage)} entri")

    # Filter: bukan GSS berlabel (title_gss != True) + ml_pred = GSS + confidence cukup
    candidates = []
    for r in all_rows:
        if r.get("title_gss") == "True":
            continue  # sudah berlabel, bukan temuan baru
        if r.get("ml_pred") != "GSS":
            continue
        try:
            conf = float(r.get("confidence", 0))
        except ValueError:
            continue
        if conf < min_confidence:
            continue
        candidates.append(r)

    # Urutkan confidence desc
    candidates.sort(key=lambda r: -float(r.get("confidence", 0)))

    # Tulis output
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES_OUT, extrasaction="ignore")
        w.writeheader()
        for r in candidates:
            key = r.get("pdf_full_path", r.get("pdf_path", ""))
            prev = existing_triage.get(key, {})
            w.writerow({
                **r,
                "manual_review_result": prev.get("manual_review_result", ""),
                "manual_notes":         prev.get("manual_notes", ""),
            })

    # Statistik
    by_issuer  = Counter(r["issuer"]   for r in candidates)
    by_class   = Counter(r["ml_class"] for r in candidates)
    confirmed  = sum(1 for r in candidates
                     if existing_triage.get(
                         r.get("pdf_full_path", r.get("pdf_path","")), {}
                     ).get("manual_review_result","") == "confirmed")

    print(f"\n=== KANDIDAT GSS TIDAK BERLABEL ===")
    print(f"  Ambang confidence >= {min_confidence}")
    print(f"  Total kandidat     : {len(candidates)}")
    print(f"  Sudah dikonfirmasi : {confirmed}")
    print(f"\n  Per kelas ML:")
    for cls, cnt in sorted(by_class.items(), key=lambda x: -x[1]):
        print(f"    {cls:<22}: {cnt:>3}")
    print(f"\n  Top 10 emiten (n kandidat):")
    for issuer, cnt in by_issuer.most_common(10):
        print(f"    {issuer:<10}: {cnt:>3}")

    print(f"\n  Output -> {os.path.relpath(OUT_CSV, ROOT)}")
    print(f"\n--- LANGKAH SELANJUTNYA (triase manual) ---")
    print(f"  Buka {os.path.basename(OUT_CSV)} di spreadsheet.")
    print(f"  Isi kolom 'manual_review_result': confirmed / rejected / ambiguous")
    print(f"  Fokus pada 20 baris teratas (confidence tertinggi).")
    print(f"  Setelah mengisi, jalankan ulang skrip ini untuk memperbarui statistik.")

    # Tampilkan top 5 kandidat
    print(f"\n--- TOP 5 KANDIDAT (confidence tertinggi) ---")
    for r in candidates[:5]:
        label = r.get('bond_name_approx') or r.get('pdf_filename','')
        print(f"  [{r['ml_class']}] conf={r['confidence']}  {r['issuer']}  {label[:55]}")


if __name__ == "__main__":
    main()
