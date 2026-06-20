import os, fitz, re

BASE = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification\pdf_by_content\01_prospektus_utama\0. GSS Fixed"
LP = "\\\\?\\"

print("=== GSS/ EMITEN ===")
gss_path = os.path.join(BASE, "GSS")
for em in sorted(os.listdir(gss_path)):
    em_path = os.path.join(gss_path, em)
    if os.path.isdir(em_path):
        folders = [f for f in os.listdir(em_path) if os.path.isdir(os.path.join(em_path, f))]
        print(f"\n  [{em}]")
        for fn in folders:
            print(f"    {fn[:75]}")

print("\n\n=== CEK SMII 2023-2024 (baca lebih dalam) ===")
smii_check = [
    r"GSS\SMII\20231127_SMII_Penyampaian Prospektus",
    r"GSS\SMII\20231127_SMII_Penyampaian Prospektus (KOREKSI)",
    r"GSS\SMII\20241119_SMII_Penyampaian Prospektus",
]

for rel in smii_check:
    folder = os.path.join(BASE, rel)
    if not os.path.exists(folder):
        print(f"  TIDAK ADA: {rel}")
        continue
    pdfs = sorted(
        [(os.path.getsize(os.path.join(folder, f)), f) for f in os.listdir(folder) if f.lower().endswith(".pdf")],
        reverse=True
    )
    if not pdfs:
        print(f"  {rel}: no PDF")
        continue
    # Baca pdf terbesar, 10 halaman
    pdf_path = LP + os.path.join(folder, pdfs[0][1])
    try:
        doc = fitz.open(pdf_path)
        text = "\n".join(doc[i].get_text() for i in range(min(10, len(doc))))
        doc.close()
    except Exception as e:
        print(f"  {rel}: error {e}")
        continue

    # Cari nama obligasi dan keyword GSS
    kws_found = []
    low = text.lower()
    for kw in ["green bond", "obligasi hijau", "berwawasan lingkungan", "sustainability linked",
                "terkait keberlanjutan", "obligasi keberlanjutan", "berwawasan sosial", "social bond"]:
        if kw in low:
            kws_found.append(kw)

    # Cari nama obligasi dari baris
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    bond_lines = [l for l in lines[:150] if "obligasi" in l.lower() or "sukuk" in l.lower()]

    name = rel.split("\\")[-1][:40]
    print(f"\n  {name}")
    print(f"    PDF terbesar: {pdfs[0][1]} ({pdfs[0][0]//1024}KB)")
    print(f"    GSS keywords: {kws_found}")
    if bond_lines:
        print(f"    Nama obligasi: {bond_lines[0][:180]}")
        if len(bond_lines) > 1:
            print(f"                   {bond_lines[1][:180]}")
