"""
Reorganisasi folder prospektus ke:
  GSS/<EMITEN>/   — prospektus GSS
  NonGSS/<EMITEN>/ — prospektus bukan GSS
  Review/<EMITEN>/ — tidak bisa ditentukan otomatis

Jalankan dulu dalam dry-run (DRY_RUN=True), lalu set False untuk eksekusi.
"""
import fitz, os, shutil, re

BASE = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed"
LP   = "\\\\?\\"   # Windows long-path prefix

DRY_RUN = False   # set True untuk preview tanpa pindah file

# ── Keyword GSS ──────────────────────────────────────────────────────────────
GSS_KW = [
    "green bond", "obligasi hijau", "obligasi berwawasan lingkungan",
    "sukuk hijau", "green sukuk", "efek berwawasan lingkungan",
    "social bond", "obligasi sosial", "berwawasan sosial",
    "sukuk sosial", "social sukuk", "social orange",
    "sustainability linked", "terkait keberlanjutan",
    "obligasi keberlanjutan", "sustainable bond",
    "gss",
]

# ── Override manual (hasil review) ──────────────────────────────────────────
# Folder yang sudah dipastikan secara manual (override hasil scan PDF)
FORCE_GSS = set()

FORCE_NON = {
    "20230613_BBRI_Penyampaian Bukti Iklan Prospektus Ringkas Penawaran Umum Obl",  # koran Kontan
    "20230704_BBRI_Penyampaian Bukti Iklan Informasi Tambahan dan_atau Perbaikan",  # koran Kontan 2020
    "20230718_ISSP_Penyampaian Prospektus",          # Obligasi Berkelanjutan II SPINDO (konvensional)
    "20230718_ISSP_Penyampaian Prospektus (KOREKSI)",
    "20230803_ISSP_Penyampaian Prospektus (KOREKSI)",
    "20241204_BJBR_Penyampaian Prospektus",          # Surat Berharga Perpetual
    "20250225_OPPM_Penyampaian Bukti Iklan Prospektus Ringkas Obligasi Berkelanj",  # OKI konvensional
    "20250320_OPPM_Prospektus Ringkas Penawaran Umum Berkelanjutan Obligasi Berk",  # OKI PUB II konvensional
    "20250324_OPPM_Penyampaian Bukti Iklan Pengumuman Informasi Tambahan dan_ata",  # koran mudik
    "20250625_BBRI_Penyampaian Bukti Iklan Informasi Tambahan dan_atau Perbaikan",  # koran Kontan 2025
    "20260508_ISSP_Penyampaian Prospektus Ringkas Penawaran Umum Berkelanjutan O",  # SPINDO PUB III konvensional
}


# ── Helpers ──────────────────────────────────────────────────────────────────
def emiten_from_name(folder_name: str) -> str:
    m = re.match(r"\d{8}_([A-Z]+)_", folder_name)
    return m.group(1) if m else "UNKNOWN"


def pick_main_pdf(folder_path: str) -> str | None:
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    except Exception:
        return None
    if not files:
        return None
    sized = []
    for f in files:
        fp = os.path.join(folder_path, f)
        try:
            sized.append((os.path.getsize(fp), fp))
        except Exception:
            pass
    sized.sort(reverse=True)
    big = [fp for sz, fp in sized if sz > 50_000]
    return big[0] if big else (sized[0][1] if sized else None)


def extract_text(pdf_path: str, n_pages: int = 8) -> str:
    lp_path = LP + pdf_path if not pdf_path.startswith(LP) else pdf_path
    try:
        doc = fitz.open(lp_path)
        parts = []
        for i in range(min(n_pages, len(doc))):
            parts.append(doc[i].get_text())
        doc.close()
        return " ".join(parts).lower()
    except Exception:
        return ""


def classify(folder_name: str) -> tuple[str, str]:
    """Return ('GSS'|'NonGSS'|'Review', reason)."""
    if folder_name in FORCE_GSS:
        return "GSS", "manual override"
    if folder_name in FORCE_NON:
        return "NonGSS", "manual override"

    folder_path = os.path.join(BASE, folder_name)
    pdf = pick_main_pdf(folder_path)
    if not pdf:
        return "Review", "tidak ada PDF"

    text = extract_text(pdf)
    if not text.strip():
        return "Review", "gagal baca teks (image PDF?)"

    for kw in GSS_KW:
        if kw in text:
            return "GSS", f"keyword: '{kw}'"

    return "NonGSS", "tidak ada keyword GSS"


def move_folder(folder_name: str, category: str):
    import time, gc
    emiten = emiten_from_name(folder_name)
    src = os.path.join(BASE, folder_name)
    dst_dir = os.path.join(BASE, category, emiten)
    dst = os.path.join(dst_dir, folder_name)
    if DRY_RUN:
        print(f"  [dry] mv {folder_name!r}")
        print(f"         -> {category}/{emiten}/")
    else:
        os.makedirs(dst_dir, exist_ok=True)
        gc.collect()
        for attempt in range(5):
            try:
                shutil.move(src, dst)
                print(f"  [mv] {folder_name[:60]}")
                print(f"       -> {category}/{emiten}/")
                return
            except (PermissionError, OSError) as e:
                if attempt < 4:
                    time.sleep(2)
                else:
                    print(f"  [!] Gagal pindah {folder_name[:50]}: {e}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    all_folders = sorted([
        d for d in os.listdir(BASE)
        if os.path.isdir(os.path.join(BASE, d))
        and d not in ("GSS", "NonGSS", "Review")
    ])

    print(f"Total folder: {len(all_folders)}")
    print(f"Mode: {'DRY RUN' if DRY_RUN else 'EKSEKUSI'}\n")

    results = {"GSS": [], "NonGSS": [], "Review": []}

    for name in all_folders:
        cat, reason = classify(name)
        emiten = emiten_from_name(name)
        results[cat].append((name, emiten, reason))
        tag = {"GSS": "[GSS]", "NonGSS": "[NON]", "Review": "[?]"}[cat]
        print(f"{tag} {name[:60]}")
        print(f"     {emiten} | {reason}")

    print(f"\n{'='*60}")
    print(f"GSS    : {len(results['GSS'])}")
    print(f"NonGSS : {len(results['NonGSS'])}")
    print(f"Review : {len(results['Review'])}")

    print("\nGSS per emiten:")
    from collections import Counter
    c = Counter(e for _, e, _ in results["GSS"])
    for em, n in sorted(c.items()):
        print(f"  {em}: {n}")

    print("\nNonGSS per emiten:")
    c2 = Counter(e for _, e, _ in results["NonGSS"])
    for em, n in sorted(c2.items()):
        print(f"  {em}: {n}")

    if results["Review"]:
        print("\nReview (perlu cek manual):")
        for name, em, reason in results["Review"]:
            print(f"  {em} | {name[:60]} | {reason}")

    if not DRY_RUN:
        confirm = input("\nLanjutkan pindah semua folder? [y/N] ").strip().lower()
        if confirm != "y":
            print("Dibatalkan.")
            return

    print("\nMemindahkan...")
    for cat, items in results.items():
        for name, emiten, _ in items:
            move_folder(name, cat)

    print("\nSelesai.")


if __name__ == "__main__":
    main()
