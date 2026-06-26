import fitz, os

BASE = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed"
LP_PREFIX = "\\\\?\\"

NON_GSS = [
    "20230613_BBRI_Penyampaian Bukti Iklan Prospektus Ringkas Penawaran Umum Obl",
    "20230704_BBRI_Penyampaian Bukti Iklan Informasi Tambahan dan_atau Perbaikan",
    "20230718_ISSP_Penyampaian Prospektus",
    "20230718_ISSP_Penyampaian Prospektus (KOREKSI)",
    "20230803_ISSP_Penyampaian Prospektus (KOREKSI)",
    "20230929_SMFP_Penyampaian Prospektus",
    "20231127_SMII_Penyampaian Prospektus",
    "20231127_SMII_Penyampaian Prospektus (KOREKSI)",
    "20240229_PNMP_Penyampaian Prospektus",
    "20240308_SMFP_Penyampaian Prospektus",
    "20240611_BJBR_Penyampaian Bukti Iklan Prospektus Penawaran Umum Berkelanjut",
    "20241119_SMII_Penyampaian Prospektus",
    "20241204_BJBR_Penyampaian Prospektus",
    "20250225_OPPM_Penyampaian Bukti Iklan Prospektus Ringkas Obligasi Berkelanj",
    "20250226_OPPM_Penyampaian Bukti Iklan Pengumuman Informasi Tambahan dan_ata",
    "20250311_SMII_Penyampaian Prospektus",
    "20250320_OPPM_Prospektus Ringkas Penawaran Umum Berkelanjutan Obligasi Berk",
    "20250324_OPPM_Penyampaian Bukti Iklan Pengumuman Informasi Tambahan dan_ata",
    "20250416_ISSP_Penyampaian Prospektus",
    "20250625_BBRI_Penyampaian Bukti Iklan Informasi Tambahan dan_atau Perbaikan",
    "20250815_PPGD_Penyampaian Prospektus",
    "20250827_PPGD_Penyampaian Prospektus",
    "20251020_IIFF_Penyampaian Prospektus",
    "20251105_SMII_Penyampaian Prospektus",
    "20260429_IIFF_Penyampaian Prospektus",
    "20260508_ISSP_Penyampaian Prospektus Ringkas Penawaran Umum Berkelanjutan O",
    "20260605_ISSP_Penyampaian Prospektus",
]


def get_bond_info(folder_name):
    folder = os.path.join(BASE, folder_name)
    try:
        files = [f for f in os.listdir(folder) if f.lower().endswith(".pdf")]
    except Exception as e:
        return f"(folder error: {e})"
    if not files:
        return "(no pdf)"

    files_with_size = []
    for f in files:
        fp = os.path.join(folder, f)
        try:
            sz = os.path.getsize(fp)
            files_with_size.append((sz, f))
        except:
            pass
    files_with_size.sort(reverse=True)
    big = [f for sz, f in files_with_size if sz > 50_000]
    chosen = big[0] if big else (files_with_size[0][1] if files_with_size else None)
    if not chosen:
        return "(no suitable pdf)"

    pdf_path = LP_PREFIX + os.path.join(folder, chosen)
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i in range(min(4, len(doc))):
            text += doc[i].get_text()
        doc.close()

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        relevant = []
        for l in lines[:100]:
            low = l.lower()
            if any(k in low for k in ["obligasi", "sukuk", "efek", "bond", "mtn",
                                       "surat utang", "surat berharga", "penawaran umum"]):
                relevant.append(l)
                if len(relevant) >= 4:
                    break
        return " | ".join(relevant) if relevant else (lines[0][:150] if lines else "(kosong)")
    except Exception as e:
        return f"(error: {e})"


for name in NON_GSS:
    info = get_bond_info(name)
    print(f"{name}")
    print(f"  -> {info[:200]}")
    print()
