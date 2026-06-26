"""
Engine klasifikasi GSS — INTI Analytical Note (rule-based, taxonomy-grounded).

Mempromosikan logika demo_classify.py menjadi modul yang dapat dipakai ulang &
diuji. Berbeda dari demo, engine ini MEMAKAI sinyal yang sudah didefinisikan di
taxonomy.py namun belum terpakai:
  - Level 0 (struktur instrumen): deteksi Sustainability-Linked & Sukuk Wakaf
  - NEGATION_HINTS: tekan false-positive saat keyword muncul dalam konteks negasi

Alur (sesuai Kerangka §7 / skema bertingkat taxonomy.py):
  teks prospektus
    -> Level 0  : SL / Wakaf?  (cek struktur, bukan penggunaan dana)
    -> Level 1  : ekstrak bagian Use-of-Proceeds -> cocokkan ke kategori eligible
       env saja -> GREEN | sosial saja -> SOCIAL | campuran -> SUSTAINABILITY
       (+ sub-tema BLUE bila ada kata kunci kelautan)
    -> tak ada sektor terpenuhi -> NON_GSS

Prinsip: explainable by design — setiap keputusan membawa kriteria/keyword bukti.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache

from .taxonomy import (
    GSSClass, Bucket, Category,
    ALL_CATEGORIES,
    SUSTAINABILITY_LINKED_SIGNALS, WAKAF_SIGNALS, NEGATION_HINTS,
)

LP = "\\\\?\\"  # prefix Windows long-path (>260 char) — lihat CLAUDE.md

# Anchor bagian "Penggunaan Dana" (paling diskriminatif). Urut dari paling spesifik.
UOP_ANCHORS: tuple[str, ...] = (
    "rencana penggunaan dana", "penggunaan dana hasil", "penggunaan dana",
    "use of proceeds", "penggunaan hasil", "dana hasil penawaran",
)


# ---------------------------------------------------------------------------
# Pembacaan PDF (long-path safe)
# ---------------------------------------------------------------------------

def read_pdf_text(path: str, max_pages: int = 50) -> str:
    r"""Ekstrak teks max_pages pertama. Coba prefix \\?\ dulu (path panjang)."""
    import fitz  # PyMuPDF
    # (max_pages pertama; coba prefix long-path lebih dulu)
    for p in (LP + path, path):
        try:
            doc = fitz.open(p)
            try:
                t = "\n".join(doc[i].get_text() for i in range(min(max_pages, len(doc))))
            finally:
                doc.close()
            if t.strip():
                return t
        except Exception:
            continue
    return ""


def read_pdf_bytes(data: bytes, max_pages: int = 50) -> str:
    """Ekstrak teks dari PDF in-memory (mis. file unggahan web app)."""
    import fitz  # PyMuPDF
    try:
        doc = fitz.open(stream=data, filetype="pdf")
        try:
            return "\n".join(doc[i].get_text() for i in range(min(max_pages, len(doc))))
        finally:
            doc.close()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Ekstraksi bagian Use-of-Proceeds
# ---------------------------------------------------------------------------

def _count_category_kw(segment_low: str) -> int:
    return sum(1 for c in ALL_CATEGORIES for kw in c.keywords if kw in segment_low)


def find_use_of_proceeds(text: str, window: int = 6000) -> tuple[str, str]:
    """Kembalikan (segmen, anchor). Bila anchor muncul berkali-kali (mis. di
    daftar isi lalu di bab isi), pilih window dengan kata-kunci kategori
    TERBANYAK — otomatis melewati entri daftar isi yang kosong sinyal."""
    low = text.lower()
    for a in UOP_ANCHORS:                      # coba anchor paling spesifik dulu
        positions = []
        start = 0
        while True:
            idx = low.find(a, start)
            if idx == -1:
                break
            positions.append(idx)
            start = idx + len(a)
        if positions:
            # ambil kemunculan TERAKHIR: bab "Penggunaan Dana" yang sesungguhnya,
            # bukan entri daftar isi (yang muncul lebih awal & kosong sinyal).
            idx = positions[-1]
            return text[idx: idx + window], a
    return text[:window], "(anchor tidak ditemukan)"


# ---------------------------------------------------------------------------
# Ekstraksi seri instrumen dari teks prospektus
# ---------------------------------------------------------------------------

_SERIES_RE = re.compile(
    r'\bSeri(?:es)?\s+([A-Z])(?:\s+(?:dan|and)\s+(?:Seri(?:es)?\s+)?([A-Z]))*\b'
)


def extract_series(text: str, head_chars: int = 3000) -> list[str]:
    """Ekstrak designator seri (A, B, C, …) dari halaman sampul prospektus.

    Menangani: 'Seri A', 'Seri B', 'Seri A dan B', 'Seri A dan Seri B'.
    Dibatasi ke head_chars pertama untuk menghindari false-match di laporan keuangan."""
    head = text[:head_chars]
    found: list[str] = []
    for m in _SERIES_RE.finditer(head):
        # Group 1 selalu ada; group 2 hadir bila pola "dan X"
        for g in m.groups():
            if g and g not in found:
                found.append(g)
    return found


# ---------------------------------------------------------------------------
# Pencocokan kategori (Level 1) dengan guard negasi
# ---------------------------------------------------------------------------

def _is_negated(low: str, kw_pos: int, span: int = 80) -> bool:
    """True bila ada NEGATION_HINTS dalam ~span char sebelum keyword."""
    pre = low[max(0, kw_pos - span): kw_pos]
    return any(h in pre for h in NEGATION_HINTS)


def match_categories(segment: str) -> list[tuple[Category, list[str]]]:
    """Cocokkan segmen ke kategori eligible. Return [(Category, [keyword bukti tak-ternegasi])]."""
    low = segment.lower()
    hits: list[tuple[Category, list[str]]] = []
    for cat in ALL_CATEGORIES:
        found: list[str] = []
        for kw in cat.keywords:
            pos = low.find(kw)
            if pos != -1 and not _is_negated(low, pos):
                found.append(kw)
        if found:
            hits.append((cat, found))
    return hits


# ---------------------------------------------------------------------------
# Level 0 — struktur instrumen (SL / Wakaf)
# ---------------------------------------------------------------------------

_WAKAF_STRONG = ("sukuk wakaf", "cash waqf", "cwls", "ikrar wakaf", "wakaf uang", "uang wakaf")
_SL_EXPLICIT = ("sustainability-linked", "sustainability linked", "terkait keberlanjutan")
_SPT_TERMS = ("sustainability performance target", "target kinerja keberlanjutan",
              "key performance indicator", "indikator kinerja utama")
_STEP_TERMS = ("step-up", "step up", "coupon step-up", "kenaikan kupon",
               "penyesuaian tingkat bunga", "penyesuaian kupon", "margin ratchet")


def detect_level0(text: str) -> tuple[GSSClass | None, list[str]]:
    """Deteksi struktur instrumen sebelum melihat penggunaan dana. Sinyal kuat &
    spesifik -> precision tinggi. Return (kelas Level-0 atau None, bukti)."""
    low = text.lower()
    wk = [s for s in _WAKAF_STRONG if s in low]
    if wk:
        return GSSClass.WAKAF, wk
    sl = [s for s in _SL_EXPLICIT if s in low]
    if sl:
        return GSSClass.SUSTAINABILITY_LINKED, sl
    spt = [s for s in _SPT_TERMS if s in low]
    step = [s for s in _STEP_TERMS if s in low]
    if spt and step:                       # KPI/SPT + mekanisme step-up bersamaan
        return GSSClass.SUSTAINABILITY_LINKED, spt + step
    return None, []


# ---------------------------------------------------------------------------
# Hasil klasifikasi
# ---------------------------------------------------------------------------

@dataclass
class Result:
    gss_class: GSSClass
    env_sectors: list[tuple[Category, list[str]]] = field(default_factory=list)
    soc_sectors: list[tuple[Category, list[str]]] = field(default_factory=list)
    level0_evidence: list[str] = field(default_factory=list)
    anchor: str = ""
    confidence: float = 0.0

    @property
    def is_gss(self) -> bool:
        return self.gss_class != GSSClass.NON_GSS

    @property
    def n_sectors(self) -> int:
        return len(self.env_sectors) + len(self.soc_sectors)

    def sector_keys(self) -> list[str]:
        return [c.key for c, _ in self.env_sectors] + [c.key for c, _ in self.soc_sectors]


def classify(text: str) -> Result | None:
    """Klasifikasi satu dokumen. None bila teks kosong (kemungkinan PDF gambar/OCR)."""
    if not text or not text.strip():
        return None

    lvl0, l0ev = detect_level0(text)
    seg, anchor = find_use_of_proceeds(text)
    hits = match_categories(seg)
    env = [(c, kw) for c, kw in hits if c.bucket == Bucket.ENVIRONMENTAL]
    soc = [(c, kw) for c, kw in hits if c.bucket == Bucket.SOCIAL]

    if lvl0 is not None:
        gss = lvl0
    elif env and soc:
        gss = GSSClass.SUSTAINABILITY
    elif env:
        gss = GSSClass.GREEN
    elif soc:
        gss = GSSClass.SOCIAL
    else:
        gss = GSSClass.NON_GSS

    # Confidence: heuristik transparan (bukan probabilitas terkalibrasi).
    n_sect = len(hits)
    n_kw = sum(len(kw) for _, kw in hits)
    if lvl0 is not None:
        conf = 0.90
    elif gss == GSSClass.NON_GSS:
        conf = 0.35 if anchor.startswith("(") else 0.65
    else:
        conf = min(0.95, 0.45 + 0.12 * n_sect + 0.02 * n_kw)

    return Result(
        gss_class=gss, env_sectors=env, soc_sectors=soc,
        level0_evidence=l0ev, anchor=anchor, confidence=round(conf, 2),
    )
