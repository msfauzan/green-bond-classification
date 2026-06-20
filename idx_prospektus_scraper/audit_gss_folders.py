"""
Audit isi PDF di setiap folder GSS dan NonGSS.
Tampilkan nama obligasi + tipe GSS yang terdeteksi.
"""
import fitz, os, re

BASE = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed"
LP   = "\\\\?\\"

GSS_PATTERNS = {
    "Green Bond"         : [r"green bond", r"obligasi hijau", r"obligasi berwawasan lingkungan", r"sukuk hijau", r"green sukuk"],
    "Social Bond"        : [r"social bond", r"obligasi sosial", r"berwawasan sosial", r"sukuk sosial", r"social orange"],
    "Sustainability-Linked": [r"sustainability.linked", r"terkait keberlanjutan", r"obligasi keberlanjutan", r"obligasi terkait keberlanjutan"],
    "Sustainable Bond"   : [r"sustainable bond", r"obligasi berkelanjutan berwawasan"],
}

BOND_NAME_HINTS = [
    r"obligasi(?:\s+\w+){0,6}(?:green|hijau|sosial|social|keberlanjutan|berwawasan|sustainability)",
    r"sukuk(?:\s+\w+){0,4}(?:green|hijau|sosial|keberlanjutan)",
    r"(?:green|social|sustainability.linked)\s+bond(?:\s+\w+){0,6}",
    r"obligasi(?:\s+\w+){1,8}(?:tahun\s+\d{4})",
]


def extract_text(pdf_path: str, n_pages: int = 5) -> str:
    try:
        lp = LP + pdf_path if not pdf_path.startswith(LP) else pdf_path
        doc = fitz.open(lp)
        parts = [doc[i].get_text() for i in range(min(n_pages, len(doc)))]
        doc.close()
        return "\n".join(parts)
    except Exception:
        return ""


def pick_pdf(folder_path: str) -> str | None:
    try:
        files = [(os.path.getsize(os.path.join(folder_path, f)), f)
                 for f in os.listdir(folder_path)
                 if f.lower().endswith(".pdf")]
        files.sort(reverse=True)
        big = [f for sz, f in files if sz > 80_000]
        return os.path.join(folder_path, big[0]) if big else (
               os.path.join(folder_path, files[0][1]) if files else None)
    except Exception:
        return None


def detect_gss_type(text: str) -> str:
    low = text.lower()
    found = []
    for gss_type, patterns in GSS_PATTERNS.items():
        for pat in patterns:
            if re.search(pat, low):
                found.append(gss_type)
                break
    return ", ".join(found) if found else "—"


def extract_bond_name(text: str) -> str:
    """Cari nama obligasi dari baris-baris teks."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Cari baris dengan kata kunci nama obligasi
    candidates = []
    for line in lines[:100]:
        low = line.lower()
        if any(k in low for k in [
            "obligasi", "sukuk", "green bond", "social bond",
            "berwawasan", "keberlanjutan", "sustainability"
        ]) and len(line) > 15:
            candidates.append(line)
            if len(candidates) >= 4:
                break
    if candidates:
        # Ambil yang paling panjang/informatif
        return max(candidates, key=len)[:200]
    return lines[0][:100] if lines else "(kosong)"


def audit_category(category: str):
    cat_path = os.path.join(BASE, category)
    if not os.path.exists(cat_path):
        return
    print(f"\n{'='*70}")
    print(f"  {category}/")
    print(f"{'='*70}")

    for emiten in sorted(os.listdir(cat_path)):
        em_path = os.path.join(cat_path, emiten)
        if not os.path.isdir(em_path):
            continue
        folders = sorted(os.listdir(em_path))
        print(f"\n  [{emiten}] — {len(folders)} folder")
        for folder_name in folders:
            folder_path = os.path.join(em_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            date_part = folder_name[:8]
            pdf = pick_pdf(folder_path)
            if not pdf:
                print(f"    {date_part} | (no PDF)")
                continue
            text = extract_text(pdf)
            if not text.strip():
                print(f"    {date_part} | (image PDF, tidak bisa dibaca)")
                continue
            gss_type = detect_gss_type(text)
            bond_name = extract_bond_name(text)
            print(f"    {date_part} | {gss_type}")
            print(f"           Nama: {bond_name}")


def main():
    print("AUDIT PROSPEKTUS GSS")
    print("="*70)
    audit_category("GSS")
    print()
    audit_category("NonGSS")
    print()
    audit_category("Review")


if __name__ == "__main__":
    main()
