"""
DEMO: jalankan classifier rule-based (taxonomy.py) ke satu prospektus GSS
terverifikasi, lalu tampilkan 'sektor eligible terpenuhi' + bukti tekstual.

Membuktikan konsep: nama bond cuma bilang 'green/social', tapi pipeline
mengekstrak SEKTOR mana yang didanai (untuk statistik) dan apakah klaim
tersubstansiasi (untuk screening greenwashing).
"""
import os, sys, re
import fitz  # PyMuPDF

sys.path.insert(0, r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification")
from classifier.taxonomy import (
    GREEN_CATEGORIES, SOCIAL_CATEGORIES, ALL_CATEGORIES,
    BLUE_KEYWORDS, SUSTAINABILITY_LINKED_SIGNALS, WAKAF_SIGNALS, Bucket,
)

BASE = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed\GSS"
LP = "\\\\?\\"


def pick_pdf(folder):
    files = [(os.path.getsize(os.path.join(folder, f)), f)
             for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    files.sort(reverse=True)
    return os.path.join(folder, files[0][1]) if files else None


def read_pdf(path, n=40):
    for p in [LP + path, path]:
        try:
            doc = fitz.open(p)
            t = "\n".join(doc[i].get_text() for i in range(min(n, len(doc))))
            doc.close()
            if t.strip():
                return t
        except Exception:
            continue
    return ""


def find_use_of_proceeds(text):
    """Cari bagian 'Penggunaan Dana / Use of Proceeds' — bagian paling diskriminatif."""
    low = text.lower()
    anchors = ["penggunaan dana", "rencana penggunaan dana", "use of proceeds",
               "penggunaan hasil", "dana hasil"]
    for a in anchors:
        idx = low.find(a)
        if idx != -1:
            return text[idx: idx + 4000], a
    return text[:4000], "(anchor tidak ketemu, pakai awal dok)"


def match_categories(segment):
    """Cocokkan teks ke kategori eligible. Return list (Category, [keyword bukti])."""
    low = segment.lower()
    hits = []
    for cat in ALL_CATEGORIES:
        found = [kw for kw in cat.keywords if kw in low]
        if found:
            hits.append((cat, found))
    blue = [kw for kw in BLUE_KEYWORDS if kw in low]
    return hits, blue


def classify(folder_path):
    pdf = pick_pdf(folder_path)
    if not pdf:
        return None
    text = read_pdf(pdf)
    if not text.strip():
        return {"status": "image PDF (perlu OCR)"}

    seg, anchor = find_use_of_proceeds(text)
    hits, blue = match_categories(seg)

    env = [(c, kw) for c, kw in hits if c.bucket == Bucket.ENVIRONMENTAL]
    soc = [(c, kw) for c, kw in hits if c.bucket == Bucket.SOCIAL]

    if env and soc:
        kelas = "SUSTAINABILITY (campuran lingkungan + sosial)"
    elif env:
        kelas = "GREEN"
    elif soc:
        kelas = "SOCIAL"
    else:
        kelas = "?? tidak ada sektor eligible terpenuhi -> RED FLAG / refinancing umum?"

    return {
        "status": "ok",
        "anchor": anchor,
        "kelas": kelas,
        "env": env,
        "soc": soc,
        "blue": blue,
        "n_sektor": len(hits),
    }


# Pilih beberapa contoh lintas tipe
SAMPLES = [
    ("BMRI", None),   # green
    ("BBRI", None),   # green + social
    ("PPGD", None),   # social
    ("PNMP", None),   # social orange
    ("BBTN", None),   # social (perumahan)
]


def report(label, r):
    print(f"\n  >> {label}")
    if not r:
        print("     (no PDF)")
        return
    if r["status"] != "ok":
        print(f"     {r['status']}")
        return
    print(f"     Anchor use-of-proceeds: '{r['anchor']}'")
    print(f"     KELAS         : {r['kelas']}")
    if r["env"]:
        print(f"     Sektor LINGKUNGAN terpenuhi:")
        for c, kw in r["env"]:
            print(f"        + {c.name_id:42} bukti: {kw[:3]}")
    if r["soc"]:
        print(f"     Sektor SOSIAL terpenuhi:")
        for c, kw in r["soc"]:
            print(f"        + {c.name_id:42} bukti: {kw[:3]}")
    if r["blue"]:
        print(f"     Sub-tema BLUE: {r['blue'][:4]}")


def run_one(em):
    ep = os.path.join(BASE, em)
    if not os.path.isdir(ep):
        print(f"  {em}: folder tidak ada")
        return
    entries = sorted(os.listdir(ep))
    subdirs = [e for e in entries if os.path.isdir(os.path.join(ep, e))]
    has_pdf = any(e.lower().endswith(".pdf") for e in entries)
    if subdirs:                                  # struktur nested
        report(f"{em} / {subdirs[0][:48]}", classify(os.path.join(ep, subdirs[0])))
    elif has_pdf:                                # struktur flat
        report(f"{em} (flat)", classify(ep))


if __name__ == "__main__":
    print("=" * 70)
    print("  DEMO CLASSIFIER RULE-BASED -> 'sektor eligible terpenuhi'")
    print("=" * 70)
    for em, _ in SAMPLES:
        run_one(em)
    print("\n" + "=" * 70)
