"""Schedule locked files for deletion on reboot, cleanup nested folders."""
import os, ctypes, ctypes.wintypes

LP = "\\\\?\\"
BASE = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed"

# MoveFileEx via ctypes
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
MOVEFILE_DELAY_UNTIL_REBOOT = 4

def schedule_delete(path: str):
    lp_path = LP + path if not path.startswith(LP) else path
    ok = kernel32.MoveFileExW(lp_path, None, MOVEFILE_DELAY_UNTIL_REBOOT)
    if ok:
        print(f"[sched] {os.path.basename(path)}")
    else:
        err = ctypes.get_last_error()
        # try without LP
        ok2 = kernel32.MoveFileExW(path, None, MOVEFILE_DELAY_UNTIL_REBOOT)
        if ok2:
            print(f"[sched] {os.path.basename(path)}")
        else:
            print(f"[!] gagal schedule {os.path.basename(path)}: err={err}")


# Files to schedule for deletion
locked = [
    r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed\20230613_BBRI_Penyampaian Bukti Iklan Prospektus Ringkas Penawaran Umum Obl\20230613_BBRI_Penyampaian Bukti Iklan_31330617_lamp3.pdf",
    r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed\20260227_BBRI_Penyampaian Prospektus\20260227_BBRI_Penyampaian Prospektus_Informasi Tambahan_32029855_lamp1.pdf",
    r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed\20260605_ISSP_Penyampaian Prospektus\20260605_ISSP_Penyampaian Prospektus_Informasi Tambahan_32097561_lamp1.pdf",
    r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed\20260605_ISSP_Penyampaian Prospektus\20260605_ISSP_Penyampaian Prospektus_Informasi Tambahan_32097561_lamp2.pdf",
]

# Cleanup nested folders (ISSP)
nested_issp = os.path.join(BASE, "NonGSS", "ISSP",
    "20260605_ISSP_Penyampaian Prospektus",
    "20260605_ISSP_Penyampaian Prospektus")

print("=== Schedule delete locked files ===")
for f in locked:
    schedule_delete(f)

print("\n=== Cleanup nested folders ===")
if os.path.exists(nested_issp):
    contents = os.listdir(nested_issp)
    print(f"ISSP nested contents: {contents}")
    if not contents:
        os.rmdir(nested_issp)
        print("[rm] ISSP nested kosong, dihapus")
    else:
        # Schedule semua files di nested untuk dihapus
        for f in contents:
            fp = os.path.join(nested_issp, f)
            if os.path.isfile(fp):
                schedule_delete(fp)
        # Schedule nested folder itu sendiri (dengan semua isinya)
        print("[!] ISSP nested tidak kosong, file di-schedule")
else:
    print("ISSP nested: tidak ada (sudah bersih)")

print("\n=== Selesai ===")
print("File-file duplikat akan otomatis terhapus saat Windows reboot berikutnya.")
print("3 folder sumber kosong bisa dihapus manual setelah reboot:")
print("  - 20230613_BBRI_Penyampaian Bukti Iklan Prospektus Ringkas Penawaran Umum Obl")
print("  - 20260227_BBRI_Penyampaian Prospektus")
print("  - 20260605_ISSP_Penyampaian Prospektus")
