"""
Cek detail untuk kasus-kasus ambigu:
1. BJBR di GSS - apakah benar GSS?
2. SMII 2023-2024 - nama obligasi lengkap dan konteks 'green bond'
3. NonGSS yang perlu diverifikasi (PPGD Aug, SMII Mar/Nov, IIFF, PNMP Feb)
"""
import fitz, os, re

BASE = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed"
LP = "\\\\?\\"

CASES = {
    "BJBR di GSS": [
        r"GSS\BJBR\20241204_BJBR_Penyampaian Bukti Iklan Prospektus Ringkas dalam rangka Penaw",
    ],
    "SMII 2023-2024 (GSS - perlu verifikasi nama)": [
        r"GSS\SMII\20231127_SMII_Penyampaian Prospektus",
        r"GSS\SMII\20241119_SMII_Penyampaian Prospektus",
    ],
    "NonGSS - PPGD Aug 2025 (sebelum GSS era?)": [
        r"NonGSS\PPGD\20250815_PPGD_Penyampaian Prospektus",
        r"NonGSS\PPGD\20250827_PPGD_Penyampaian Prospektus",
    ],
    "NonGSS - SMII (konvensional atau GSS?)": [
        r"NonGSS\SMII\20250311_SMII_Penyampaian Prospektus",
        r"NonGSS\SMII\20251105_SMII_Penyampaian Prospektus",
    ],
    "NonGSS - IIFF": [
        r"NonGSS\IIFF\20251020_IIFF_Penyampaian Prospektus",
        r"NonGSS\IIFF\20260429_IIFF_Penyampaian Prospektus",
    ],
    "NonGSS - PNMP Feb 2024": [
        r"NonGSS\PNMP\20240229_PNMP_Penyampaian Prospektus",
    ],
    "Review folders": [
        r"Review\BBNI\20250612_BBNI_Penyampaian Bukti Iklan Prospektus Ringkas dal",
        r"Review\BJBR\20241204_BJBR_Penyampaian Bukti Iklan Prospektus Ringkas dal",
        r"Review\OPPM\20231002_OPPM_Penyampaian Bukti Iklan Prospektus Ringkas Pen",
        r"Review\OPPM\20250320_OPPM_Penyampaian Bukti Iklan Informasi Tambahan dan",
    ],
}


def pick_pdf(folder: str):
    try:
        files = [(os.path.getsize(os.path.join(folder, f)), f)
                 for f in os.listdir(folder) if f.lower().endswith(".pdf")]
        files.sort(reverse=True)
        big = [f for sz, f in files if sz > 50_000]
        return os.path.join(folder, big[0] if big else files[0][1]) if files else None
    except Exception:
        return None


def read_pdf(pdf_path: str, n_pages: int = 12) -> str:
    for prefix in [LP + pdf_path, pdf_path]:
        try:
            doc = fitz.open(prefix)
            t = "\n".join(doc[i].get_text() for i in range(min(n_pages, len(doc))))
            doc.close()
            if t.strip():
                return t
        except Exception:
            continue
    return ""


def analyze(folder_path: str, label: str):
    if not os.path.exists(folder_path):
        print(f"  TIDAK ADA: {label}")
        return

    pdf = pick_pdf(folder_path)
    if not pdf:
        print(f"  {label}: no PDF")
        return

    text = read_pdf(pdf)
    if not text.strip():
        print(f"  {label}: image PDF (tidak bisa dibaca)")
        return

    low = text.lower()
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # Deteksi GSS
    gss_hits = []
    for kw in ["green bond", "obligasi hijau", "berwawasan lingkungan", "sukuk hijau",
                "social bond", "obligasi sosial", "berwawasan sosial", "sukuk sosial", "social orange",
                "sustainability linked", "terkait keberlanjutan", "obligasi keberlanjutan",
                "obligasi terkait keberlanjutan"]:
        if kw in low:
            # Cari konteks di mana keyword muncul
            idx = low.find(kw)
            ctx = text[max(0, idx-50):idx+100].replace("\n", " ").strip()
            gss_hits.append(f"'{kw}' >> ...{ctx}...")
            if len(gss_hits) >= 2:
                break

    # Cari nama seri obligasi
    bond_name_lines = []
    for l in lines[:200]:
        low_l = l.lower()
        if any(k in low_l for k in ["obligasi berkelanjutan", "obligasi berwawasan", "obligasi terkait",
                                      "sukuk", "penawaran umum berkelanjutan"]) and len(l) > 20:
            bond_name_lines.append(l)
            if len(bond_name_lines) >= 3:
                break

    date_part = label[:8] if label[:8].isdigit() else label[:40]
    print(f"\n  [{label[:50]}]")
    if gss_hits:
        for h in gss_hits:
            print(f"    GSS: {h[:160]}")
    else:
        print(f"    GSS: TIDAK DITEMUKAN")
    for bn in bond_name_lines:
        print(f"    Nama: {bn[:180]}")


for group, paths in CASES.items():
    print(f"\n{'='*65}")
    print(f"  {group}")
    print(f"{'='*65}")
    for rel_path in paths:
        folder = os.path.join(BASE, rel_path)
        label = os.path.basename(rel_path)
        analyze(folder, label)
