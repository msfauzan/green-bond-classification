"""
Prioritaskan emiten untuk di-scrape prospektusnya.

Logika:
  1. Baca universe_title_census.csv (output Phase 1)
  2. Buang emiten gold set (sudah punya prospektus)
  3. Kelompokkan per emiten; hitung n_bond + total outstanding
  4. Tandai tier prioritas berdasarkan:
       - Tier 1: energi/utilitas/infrastruktur/air/transportasi (paling mungkin GSS)
       - Tier 2: properti, multifinance, konglomerasi-ESG, perbankan besar belum di-scrape
       - Tier 3: lainnya (consumer, telecom, mining, dll.)
     Tier ditentukan dari kode emiten (known sectors) + kata kunci nama obligasi.
  5. Output data/scraping_priority.csv, diurutkan per tier lalu outstanding desc.
     Kolom batch disarankan (~20-30 emiten per sesi scraping).

Jalankan dari root repo:
  python code/evaluation/prioritize_scraping.py
"""
from __future__ import annotations
import csv
import os
import sys
from collections import defaultdict

ROOT     = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
CODE_DIR = os.path.join(ROOT, "code")
DATA_DIR = os.path.join(ROOT, "data")

IN_CSV  = os.path.join(DATA_DIR, "universe_title_census.csv")
OUT_CSV = os.path.join(DATA_DIR, "scraping_priority.csv")

# Emiten sudah punya prospektus (gold set + scraped)
GOLD_ISSUERS = {
    "ARKO","BBNI","BBRI","BBTN","BJBR","BMRI",
    "BRIS","OPPM","PNMP","POLI","PPGD","SMII",
    "SMFP","IIFF","ISSP",  # di-scrape tapi bukan gold
}

# ---------------------------------------------------------------------------
# Tier 1 — sektor yang paling mungkin GSS (energi, infrastruktur, air, transport)
# Berdasarkan kode emiten yang dikenal di pasar Indonesia
# ---------------------------------------------------------------------------
TIER1_CODES = {
    # PLN & anak perusahaan
    "PPLN",  # PLN
    "IPAR",  # Indonesia Power (anak PLN)
    # Utilitas & energi terbarukan
    "PGEO",  # Pertamina Geothermal Energy
    "PGAS",  # PGN (gas distribusi)
    "PLTM",  # Pembangkit Listrik Tenaga Mikro (micro-hydro)
    "PLTU",  # PLTU-xxx
    "EPAC",  # energi
    "KEEN",  # Kencana Energi
    "ESSA",  # ESSA Industries (energi)
    # Air & sanitasi
    "SPAM",  # SPAM (water utility)
    "AETRA", # AETRA Air Jakarta
    "PDAM",  # PDAM-related
    # Transportasi & infrastruktur
    "JSMR",  # Jasa Marga (toll roads)
    "WIKA",  # Wijaya Karya (konstruksi)
    "PTPP",  # PP Persero
    "ADHI",  # Adhi Karya
    "WSKT",  # Waskita Karya
    "BUKAKA","BUKS",  # Bukaka
    "HUTAMA","HUTM",  # Hutama Karya
    "KIJA",  # Kawasan industri
    "JAKARTA","JAKA",
    # Pertanian / pangan berkelanjutan
    "PTPN",  # PTPN (perkebunan negara)
    "BISI",  # benih
    # Telematika (jaringan — fiber, green data center)
    "ISAT",  # Indosat
    "EXCL",  # XL Axiata
    "TLKM",  # Telkom
    "TELE",
}

# Tier 2 — properti, multifinance ESG, bank besar belum di-scrape
TIER2_CODES = {
    # Properti (green building potensial)
    "BSDE","SMRA","CTRA","LPKR","PWON","ASRI","DILD","DART","JRPT",
    "MKPI","MTLA","APLN","MDLN","KPIG","GPRA","GMTD",
    # Multifinance & lembaga keuangan pembangunan
    "ADMF","BFIN","MFIN","VRNA","CFIN","HDFA","BCAP","IMFI","MCFS","TIFA",
    "WOMF","BPFI","AMAG","BVIC","BNLI","MEGA","BNBA",
    # Bank menengah-besar belum di-scrape (ESG framework potensial)
    "BBCA","CIMB","BNGA","BNII","BDMN","PNBN","NISP","BJTM","BPTN",
    "BJBS","BSIM","AGRO","BTPN","BTPS","BMAS","BCIC","BGTG",
    # Infrastruktur / utilitas lain
    "TOTL","NRCA","DGIK","IDPR",
}

# ---------------------------------------------------------------------------
# Kata kunci nama obligasi yang menunjukkan potensi GSS (bukan label resmi)
# — dipakai untuk menaikkan tier instrumen yang lolos keyword tapi bukan emiten Tier1/2
# ---------------------------------------------------------------------------
GREEN_NAME_HINTS = (
    "energi", "energy", "hijau", "green", "iklim", "climate",
    "infrastruktur", "infrastructure", "air bersih", "sanitasi",
    "transportasi", "renewable", "terbarukan", "solar", "angin", "hidro",
    "geothermal", "lingkungan",
)
SOCIAL_NAME_HINTS = (
    "sosial", "social", "umkm", "mikro", "perumahan", "housing",
    "pangan", "ketahanan", "inklusif", "komunitas", "kesehatan", "pendidikan",
)


def _tier_from_code(issuer_code: str) -> int:
    code = issuer_code.upper()
    if code in TIER1_CODES:
        return 1
    if code in TIER2_CODES:
        return 2
    return 3


def _name_has_hint(name: str, hints: tuple[str, ...]) -> bool:
    low = name.lower()
    return any(h in low for h in hints)


def main():
    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} tidak ditemukan. Jalankan universe_title_census.py dulu.")
        sys.exit(1)

    # Baca census
    with open(IN_CSV, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # Filter: bukan GSS berlabel + bukan gold issuer
    candidates = [
        r for r in rows
        if r["is_gss_titled"] == "False"
        and r["IssuerCode"].strip() not in GOLD_ISSUERS
    ]

    print(f"Kandidat (bukan GSS berlabel, bukan gold issuer): {len(candidates):,} instrumen")

    # Kelompokkan per emiten
    issuer_stats: dict[str, dict] = defaultdict(lambda: {
        "n_bonds": 0, "total_outstanding": 0.0,
        "tier_from_code": 3, "has_green_hint": False, "has_social_hint": False,
        "sample_bonds": [],
    })

    for r in candidates:
        code = r["IssuerCode"].strip()
        name = r["BondName"].strip()
        try:
            outstanding = float(r["Outstanding"]) if r["Outstanding"] else 0.0
        except ValueError:
            outstanding = 0.0

        s = issuer_stats[code]
        s["n_bonds"] += 1
        s["total_outstanding"] += outstanding
        s["tier_from_code"] = min(s["tier_from_code"], _tier_from_code(code))
        if _name_has_hint(name, GREEN_NAME_HINTS):
            s["has_green_hint"] = True
        if _name_has_hint(name, SOCIAL_NAME_HINTS):
            s["has_social_hint"] = True
        if len(s["sample_bonds"]) < 2:
            s["sample_bonds"].append(name[:60])

    # Buat baris output + hitung tier final
    out_rows = []
    for code, s in issuer_stats.items():
        tier = s["tier_from_code"]
        # Naikkan ke tier 2 bila nama obligasi punya green/social hint
        if tier == 3 and (s["has_green_hint"] or s["has_social_hint"]):
            tier = 2
        out_rows.append({
            "IssuerCode":              code,
            "n_bonds":                 s["n_bonds"],
            "total_outstanding_triliun": round(s["total_outstanding"] / 1e12, 3),
            "priority_tier":           tier,
            "green_name_hint":         "Ya" if s["has_green_hint"] else "",
            "social_name_hint":        "Ya" if s["has_social_hint"] else "",
            "sample_bond_names":       " | ".join(s["sample_bonds"]),
        })

    # Urutkan: tier asc, outstanding desc
    out_rows.sort(key=lambda r: (r["priority_tier"], -r["total_outstanding_triliun"]))

    # Tambah kolom batch (setiap ~25 emiten per batch, dalam tier yang sama)
    batch = 1
    count_in_batch = 0
    prev_tier = None
    for r in out_rows:
        if prev_tier is not None and r["priority_tier"] != prev_tier:
            batch += 1
            count_in_batch = 0
        elif count_in_batch >= 25:
            batch += 1
            count_in_batch = 0
        r["batch"] = batch
        prev_tier = r["priority_tier"]
        count_in_batch += 1

    # Tulis CSV
    fieldnames = [
        "IssuerCode","n_bonds","total_outstanding_triliun",
        "priority_tier","batch",
        "green_name_hint","social_name_hint",
        "sample_bond_names",
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    # Ringkasan
    tier_counts = defaultdict(int)
    for r in out_rows:
        tier_counts[r["priority_tier"]] += 1

    print(f"\n=== PRIORITAS SCRAPING ===")
    for tier in (1, 2, 3):
        print(f"  Tier {tier}: {tier_counts[tier]:>3} emiten")
    print(f"  Total batch yang disarankan: {max(r['batch'] for r in out_rows)}")
    print(f"\n  Output -> {os.path.relpath(OUT_CSV, ROOT)}")
    print(f"\n--- TOP 10 PRIORITAS ---")
    for r in out_rows[:10]:
        hint = ""
        if r["green_name_hint"]: hint += "[G]"
        if r["social_name_hint"]: hint += "[S]"
        print(f"  Tier{r['priority_tier']} Batch{r['batch']:02d}  "
              f"{r['IssuerCode']:<8}  "
              f"{r['n_bonds']:>3} bonds  "
              f"Rp {r['total_outstanding_triliun']:>6.2f}T  "
              f"{hint}")

    print(f"\nContoh perintah scraping (Tier 1, Batch 1):")
    tier1_batch1 = [r["IssuerCode"] for r in out_rows
                    if r["priority_tier"] == 1 and r["batch"] == 1]
    if tier1_batch1:
        codes_str = " ".join(tier1_batch1)
        print(f"  cd code/idx_prospektus_scraper")
        print(f"  python scrape_prospektus.py --codes {codes_str} --from-date 2019-01-01")


if __name__ == "__main__":
    main()
