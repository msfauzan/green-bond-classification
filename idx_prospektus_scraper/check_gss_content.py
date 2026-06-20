"""
Scan PDF di 0. GSS Fixed, cek apakah isinya adalah prospektus GSS bond/sukuk.
Pisahkan ke subfolder: GSS/ dan NonGSS/
"""
import fitz  # PyMuPDF
import shutil
from pathlib import Path

BASE = Path(r"\\?\D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed")

GSS_KEYWORDS = [
    # Green
    "green bond", "obligasi hijau", "obligasi berwawasan lingkungan",
    "sukuk hijau", "green sukuk", "efek berwawasan lingkungan",
    # Social
    "social bond", "obligasi sosial", "berwawasan sosial",
    "sukuk sosial", "social sukuk", "social orange",
    # Sustainability linked
    "sustainability linked", "terkait keberlanjutan", "obligasi keberlanjutan",
    "sustainable bond", "obligasi berkelanjutan berwawasan",
    # Generic GSS markers
    "gss", "green, social", "hijau, sosial",
]

# Kode BEI yang GSS
GSS_BEI_CODES = {"GN", "GNCN", "SO", "SOCN", "SE", "SECN", "SL", "SLCN"}


def extract_text_first_pages(pdf_path: Path, n_pages: int = 5) -> str:
    try:
        doc = fitz.open(str(pdf_path))
        texts = []
        for i in range(min(n_pages, len(doc))):
            texts.append(doc[i].get_text())
        doc.close()
        return " ".join(texts).lower()
    except Exception as e:
        return ""


def is_gss(text: str) -> tuple[bool, str]:
    """Return (is_gss, matched_keyword)."""
    for kw in GSS_KEYWORDS:
        if kw in text:
            return True, kw
    return False, ""


def pick_main_pdf(folder: Path) -> Path | None:
    """Pilih PDF terbesar di folder (paling mungkin isi prospektus)."""
    pdfs = sorted(folder.glob("*.pdf"), key=lambda f: f.stat().st_size, reverse=True)
    if not pdfs:
        return None
    # Hindari file yang terlalu kecil (cover letter ~5KB)
    # Kalau yang terbesar > 50KB, pakai itu. Kalau tidak ada, pakai yang ada.
    big = [p for p in pdfs if p.stat().st_size > 50_000]
    return big[0] if big else pdfs[0]


def main():
    folders = sorted([d for d in BASE.iterdir() if d.is_dir()])
    print(f"Total folder: {len(folders)}\n")

    gss_folders = []
    non_gss_folders = []
    unclear = []

    for folder in folders:
        pdf = pick_main_pdf(folder)
        if pdf is None:
            print(f"[?] {folder.name}  — tidak ada PDF")
            unclear.append(folder)
            continue

        text = extract_text_first_pages(pdf, n_pages=5)
        if not text.strip():
            print(f"[?] {folder.name}  — gagal baca teks")
            unclear.append(folder)
            continue

        ok, kw = is_gss(text)
        if ok:
            print(f"[GSS] {folder.name}")
            print(f"      keyword: '{kw}'")
            gss_folders.append(folder)
        else:
            print(f"[NON] {folder.name}")
            non_gss_folders.append(folder)

    print(f"\n{'='*60}")
    print(f"GSS     : {len(gss_folders)}")
    print(f"Non-GSS : {len(non_gss_folders)}")
    print(f"Unclear : {len(unclear)}")

    if non_gss_folders or unclear:
        print("\nNon-GSS folders:")
        for f in non_gss_folders:
            print(f"  - {f.name}")
        if unclear:
            print("\nUnclear (perlu cek manual):")
            for f in unclear:
                print(f"  - {f.name}")

    # Tanya konfirmasi sebelum pindah
    if non_gss_folders:
        ans = input(f"\nPindahkan {len(non_gss_folders)} non-GSS folder ke 'NonGSS/'? [y/N] ").strip().lower()
        if ans == "y":
            non_gss_dir = BASE / "NonGSS"
            non_gss_dir.mkdir(exist_ok=True)
            for f in non_gss_folders:
                dest = non_gss_dir / f.name
                shutil.move(str(f), str(dest))
                print(f"  [mv] {f.name}")
            print(f"\nSelesai. {len(non_gss_folders)} folder dipindah ke NonGSS/")


if __name__ == "__main__":
    main()
