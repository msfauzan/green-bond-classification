"""
Sensus seluruh 1,437 EBUS korporasi IDX berdasarkan nama instrumen.

Menggunakan title_lookup.gss_type_from_title() untuk menandai setiap
instrumen — tanpa memerlukan prospektus, hanya dari nama.

Output:
  data/universe_title_census.csv   -- satu baris per instrumen
  (cetak ringkasan ke stdout)

Jalankan dari root repo:
  python code/evaluation/universe_title_census.py
"""
from __future__ import annotations
import csv
import os
import sys
from collections import Counter

ROOT     = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
CODE_DIR = os.path.join(ROOT, "code")
DATA_DIR = os.path.join(ROOT, "data")
sys.path.insert(0, CODE_DIR)

from classifier.title_lookup import all_instruments, gss_type_from_title  # noqa: E402

OUT_CSV = os.path.join(DATA_DIR, "universe_title_census.csv")


def main():
    instruments = all_instruments()
    if not instruments:
        print("ERROR: universe CSV tidak ditemukan atau kosong.")
        sys.exit(1)

    rows = []
    for inst in instruments:
        bond_id    = inst.get("BondId", "").strip()
        issuer     = inst.get("IssuerCode", "").strip()
        name       = inst.get("BondName", "").strip()
        outstanding = inst.get("Outstanding", "").strip()
        mature_date = inst.get("MatureDate", "").strip()
        rating     = inst.get("Rating", "").strip()

        gss_type   = gss_type_from_title(name)   # None jika bukan GSS
        is_gss     = gss_type is not None

        rows.append({
            "BondId":         bond_id,
            "IssuerCode":     issuer,
            "BondName":       name,
            "title_gss_type": gss_type or "",
            "is_gss_titled":  "True" if is_gss else "False",
            "Outstanding":    outstanding,
            "MatureDate":     mature_date,
            "Rating":         rating,
        })

    # Tulis CSV
    fieldnames = ["BondId","IssuerCode","BondName",
                  "title_gss_type","is_gss_titled",
                  "Outstanding","MatureDate","Rating"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Ringkasan
    n_total = len(rows)
    gss_rows = [r for r in rows if r["is_gss_titled"] == "True"]
    n_gss = len(gss_rows)
    n_non = n_total - n_gss

    type_count = Counter(r["title_gss_type"] for r in gss_rows)

    print(f"\n=== SENSUS UNIVERSE EBUS KORPORASI (berdasarkan judul) ===")
    print(f"  Total instrumen  : {n_total:,}")
    print(f"  GSS berlabel     : {n_gss:,}  ({n_gss/n_total*100:.2f}%)")
    print(f"  Tidak berlabel   : {n_non:,}  ({n_non/n_total*100:.2f}%)")
    print(f"\n  Rincian GSS berlabel per tipe:")
    for gtype in ("green", "social", "sustainability", "sustainability_linked"):
        c = type_count.get(gtype, 0)
        print(f"    {gtype:<25}: {c:>3}")
    other = sum(v for k, v in type_count.items()
                if k not in ("green","social","sustainability","sustainability_linked"))
    if other:
        print(f"    {'lainnya':<25}: {other:>3}")

    print(f"\n  Output -> {os.path.relpath(OUT_CSV, ROOT)}")

    # Cek konsistensi dgn sensus sebelumnya (82)
    if n_gss != 82:
        print(f"\n  PERHATIAN: sensus judul menemukan {n_gss} (bukan 82) GSS berlabel.")
        print(f"  Bisa jadi ada instrumen baru/matured sejak snapshot 2026-06-18.")


if __name__ == "__main__":
    main()
