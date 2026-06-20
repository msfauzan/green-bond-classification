import os, shutil, time

BASE = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed"
LP   = "\\\\?\\"
NAME = "20230613_BBRI_Penyampaian Bukti Iklan Prospektus Ringkas Penawaran Umum Obl"

src_folder  = os.path.join(BASE, NAME)
dst_folder  = os.path.join(BASE, "NonGSS", "BBRI", NAME)
nested_dst  = os.path.join(dst_folder, NAME)
misplaced   = os.path.join(BASE, "NonGSS", "BBRI", "20230613_BBRI_Penyampaian Bukti Iklan_31330617_lamp3.pdf")

print("src_folder:", os.listdir(src_folder) if os.path.exists(src_folder) else "N/A")
print("dst_folder:", os.listdir(dst_folder) if os.path.exists(dst_folder) else "N/A")
print("nested_dst:", os.listdir(nested_dst) if os.path.exists(nested_dst) else "N/A")
print("misplaced lamp3:", os.path.exists(misplaced))

# 1. Hapus nested kosong di dst
if os.path.exists(nested_dst):
    try:
        # nested mungkin punya nested lagi
        inner = os.listdir(nested_dst)
        if not inner:
            os.rmdir(nested_dst)
            print("[OK] nested_dst dihapus")
        else:
            print("[!] nested_dst tidak kosong:", inner)
            # coba rmtree dengan LP
            shutil.rmtree(LP + nested_dst)
            print("[OK] nested_dst rmtree dihapus")
    except Exception as e:
        print("[!] nested_dst error:", e)

# 2. Pindahkan misplaced lamp3 ke dst_folder
if os.path.exists(misplaced):
    lamp3_dst = os.path.join(dst_folder, "20230613_BBRI_Penyampaian Bukti Iklan_31330617_lamp3.pdf")
    if not os.path.exists(lamp3_dst):
        shutil.copy2(LP + misplaced, LP + lamp3_dst)
        os.remove(LP + misplaced)
        print("[OK] lamp3 dipindah ke dst_folder (via LP)")
    else:
        os.remove(LP + misplaced)
        print("[OK] lamp3 duplikat di NonGSS/BBRI/ dihapus")

# 3. Salin file dari src ke dst (yang belum ada)
if os.path.exists(src_folder):
    existing_dst = set(os.listdir(dst_folder))
    for f in os.listdir(src_folder):
        dst_f = os.path.join(dst_folder, f)
        src_f = os.path.join(src_folder, f)
        if f not in existing_dst:
            try:
                shutil.copy2(LP + src_f, LP + dst_f)
                print(f"[cp] {f}")
            except Exception as e:
                print(f"[!] copy {f}: {e}")
        else:
            print(f"[skip] {f}")

# 4. Hapus source folder
time.sleep(3)
try:
    os.rmdir(src_folder)
    print("[OK] src_folder dihapus (rmdir)")
except Exception as e:
    print(f"[!] rmdir src gagal: {e}")
    # coba hapus file satu per satu dulu
    for f in os.listdir(src_folder):
        fp = os.path.join(src_folder, f)
        try:
            os.remove(LP + fp)
            print(f"[rm] {f}")
        except Exception as e2:
            print(f"[!] rm {f}: {e2}")
    try:
        os.rmdir(src_folder)
        print("[OK] src_folder dihapus setelah rm files")
    except Exception as e3:
        print(f"[!] src_folder masih ada: {e3}")

print("\nFinal dst:", os.listdir(dst_folder))
