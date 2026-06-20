import os
BASE = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed"
for cat in ["GSS", "NonGSS", "Review"]:
    cat_path = os.path.join(BASE, cat)
    if not os.path.exists(cat_path):
        continue
    for em in sorted(os.listdir(cat_path)):
        em_path = os.path.join(cat_path, em)
        if not os.path.isdir(em_path):
            continue
        for folder in sorted(os.listdir(em_path)):
            folder_path = os.path.join(em_path, folder)
            if os.path.isdir(folder_path):
                try:
                    files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
                except:
                    files = ["(err)"]
                print(f"{cat}/{em}/{folder} [{len(files)} PDF]")
