"""
Lookup GSS dari nama obligasi/sukuk di listing IDX.

Nama instrumen di BEI secara resmi mencerminkan label POJK 18/2023 —
jauh lebih andal daripada framing gate berbasis teks badan prospektus.

Jebakan kritis (lihat CLAUDE.md):
  "Obligasi Berkelanjutan I/II/III"  → PUB (shelf reg.)  → BUKAN GSS
  "Obligasi Keberlanjutan ..."       → Sustainability bond → GSS
  "Obligasi Berwawasan Lingkungan"   → Green bond         → GSS
  "Obligasi Berwawasan Sosial"       → Social bond        → GSS
  "Obligasi Berlandaskan Keberlanjutan" → GSS (POJK 18)
  "Sukuk Wakaf / CWLS"               → Sukuk Wakaf        → GSS (Level 0)
"""
from __future__ import annotations
import csv
import os
from functools import lru_cache

ROOT = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
IDX_CSV = os.path.join(ROOT, "data", "idx_obligasi_sukuk_korporasi_20260618_134135.csv")

# ---------------------------------------------------------------------------
# Penanda GSS dalam nama obligasi (POSITIF)
# Urut dari paling spesifik ke paling umum
# ---------------------------------------------------------------------------
TITLE_GSS_MARKERS: tuple[str, ...] = (
    # POJK 18/2023 — label resmi OJK
    "berwawasan lingkungan",
    "berwawasan sosial",
    "berwawasan keberlanjutan",
    "berlandaskan keberlanjutan",
    "terkait keberlanjutan",          # Sustainability-Linked
    # Nama umum pasar
    "obligasi keberlanjutan",         # "Obligasi Keberlanjutan" ≠ "Berkelanjutan"
    "sukuk keberlanjutan",
    "obligasi hijau",
    "sukuk hijau",
    "obligasi sosial",
    "sukuk sosial",
    "green bond",
    "green sukuk",
    "social bond",
    "blue bond",
    # Branded/niche
    "sosial orange",                  # PNM Orange Social Bond
)

# ---------------------------------------------------------------------------
# Penanda yang BUKAN GSS meski mengandung kata mirip (NEGATIF — pengecualian)
# ---------------------------------------------------------------------------
# "Obligasi Berkelanjutan" adalah PUB (Penawaran Umum Berkelanjutan),
# istilah administrasi shelf-registration → diabaikan oleh TITLE_GSS_MARKERS
# (karena "berkelanjutan" ≠ "keberlanjutan" maupun "berwawasan *").
# Tidak perlu daftar eksplisit — cukup TITLE_GSS_MARKERS tidak memuat
# "berkelanjutan" sendirian.


@lru_cache(maxsize=1)
def _load_idx() -> dict[str, list[str]]:
    """Kembalikan dict IssuerCode -> [BondName, ...] dari CSV."""
    issuer_bonds: dict[str, list[str]] = {}
    if not os.path.exists(IDX_CSV):
        return issuer_bonds
    with open(IDX_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            code = row.get("IssuerCode", "").strip()
            name = row.get("BondName", "").strip()
            if code and name:
                issuer_bonds.setdefault(code, []).append(name)
    return issuer_bonds


def gss_titles_for(issuer: str) -> list[str]:
    """Kembalikan daftar nama obligasi/sukuk GSS milik issuer ini."""
    idx = _load_idx()
    names = idx.get(issuer, [])
    return [n for n in names if _is_gss_titled(n)]


def has_gss_title(issuer: str) -> tuple[bool | None, list[str]]:
    """
    True  = issuer punya setidaknya satu instrumen bernama GSS
    False = issuer ada di listing tapi tidak ada instrumen bernama GSS
    None  = issuer tidak ada di listing (tidak diketahui)
    """
    idx = _load_idx()
    if issuer not in idx:
        return None, []
    hits = gss_titles_for(issuer)
    return (True if hits else False), hits


def _is_gss_titled(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in TITLE_GSS_MARKERS)
