"""
Bangun data/gold_bonds_db.csv dari semua PDF di folder gold set GSS.

Satu PDF bisa mencakup beberapa seri (A dan B). Script ini:
  1. Scan setiap PDF di GSS/<ISSUER>/
  2. Ekstrak nama obligasi + seri dari halaman sampul
  3. Cross-reference ke IDX listing untuk MatureDate & Outstanding
  4. Baris yang tidak ditemukan di IDX → kemungkinan sudah jatuh tempo

Jalankan dari root repo:
    python evaluation/build_gold_db.py
"""
from __future__ import annotations
import csv, os, re, sys, datetime

ROOT = r"D:\1. Important\Work\Bank Indonesia\DSta-DSMF\Green Bond Classification"
sys.path.insert(0, ROOT)

from classifier.engine import extract_series
from classifier.title_lookup import (
    gss_type_from_title, _load_idx, TITLE_GSS_MARKERS, _is_gss_titled,
)

GOLD_GSS = os.path.join(ROOT, "pdf_by_content", "01_prospektus_utama", "0. Fix GSS", "GSS")
IDX_CSV  = os.path.join(ROOT, "data", "idx_obligasi_sukuk_korporasi_20260618_134135.csv")
OUT_CSV  = os.path.join(ROOT, "data", "gold_bonds_db.csv")
LP = "\\\\?\\"


# ---------------------------------------------------------------------------
# Baca PDF (long-path safe, hanya 5 halaman sampul)
# ---------------------------------------------------------------------------

def _read_cover(pdf_path: str, n_pages: int = 5) -> str:
    import fitz
    for p in (LP + pdf_path, pdf_path):
        try:
            doc = fitz.open(p)
            try:
                return "\n".join(doc[i].get_text() for i in range(min(n_pages, len(doc))))
            finally:
                doc.close()
        except Exception:
            continue
    return ""


# ---------------------------------------------------------------------------
# Ekstraksi nama obligasi dari cover
# ---------------------------------------------------------------------------

# Pola nama obligasi resmi Indonesia (urut dari paling spesifik)
_NAME_RE = re.compile(
    r'(?:Obligasi|Sukuk)(?:\s+\w+){1,12}(?:Tahun\s+\d{4}|Seri\s+[A-Z])',
    re.IGNORECASE,
)

# Juga tangkap label internasional
_NAME_INTL_RE = re.compile(
    r'(?:Green Bond|Social Bond|Sustainability(?:\s+\w+)? Bond|Sustainability-Linked Bond)'
    r'(?:\s+\w+){0,8}(?:Tahun\s+\d{4}|Seri\s+[A-Z]|\d{4})',
    re.IGNORECASE,
)


def extract_bond_name(text: str, head_chars: int = 4000) -> str:
    """Ambil nama obligasi terpanjang & paling informatif dari sampul."""
    head = text[:head_chars]
    candidates: list[str] = []

    for pat in (_NAME_RE, _NAME_INTL_RE):
        candidates.extend(m.group(0).strip() for m in pat.finditer(head))

    # Fallback: baris yang mengandung penanda GSS
    if not candidates:
        for line in head.splitlines():
            stripped = line.strip()
            if len(stripped) > 20 and _is_gss_titled(stripped):
                candidates.append(stripped)

    if not candidates:
        return ""

    # Pilih kandidat yang paling panjang (paling informatif)
    return max(candidates, key=len)[:250]


# ---------------------------------------------------------------------------
# Cross-reference IDX
# ---------------------------------------------------------------------------

def _build_idx_lookup() -> dict[str, list[dict]]:
    """
    Kembalikan dict IssuerCode -> [row IDX, ...] termasuk MatureDate & Outstanding.
    """
    result: dict[str, list[dict]] = {}
    if not os.path.exists(IDX_CSV):
        return result
    with open(IDX_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            code = row.get("IssuerCode", "").strip()
            if code:
                result.setdefault(code, []).append(row)
    return result


def _match_idx(issuer: str, series: str, issue_year: str,
               idx_lookup: dict) -> dict | None:
    """
    Cari baris IDX berdasarkan issuer + seri + tahun (dari filename PDF).
    Menggunakan tahun penerbitan dari nama file — lebih andal dari teks PDF.
    Cocok bila: (1) seri cocok, (2) tahun penerbitan muncul di nama IDX,
                (3) setidaknya satu penanda GSS dalam nama IDX.
    """
    rows = idx_lookup.get(issuer, [])
    if not rows:
        return None

    for row in rows:
        idx_name = row.get("BondName", "").lower()

        # Seri harus cocok bila diketahui
        if series and f"seri {series.lower()}" not in idx_name:
            continue

        # Tahun dari filename harus muncul di nama IDX
        if issue_year and issue_year not in idx_name:
            continue

        # Nama IDX harus merupakan instrumen GSS
        if not any(m in idx_name for m in TITLE_GSS_MARKERS):
            continue

        return row

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Membangun gold_bonds_db.csv …\n")
    idx_lookup = _build_idx_lookup()
    today = datetime.date.today()

    # key = (IssuerCode, BondName_IDX) untuk deduplication
    seen: dict[tuple, dict] = {}

    for issuer in sorted(os.listdir(GOLD_GSS)):
        issuer_path = os.path.join(GOLD_GSS, issuer)
        if not os.path.isdir(issuer_path):
            continue

        pdfs = sorted(
            f for f in os.listdir(issuer_path) if f.lower().endswith(".pdf")
        )
        print(f"[{issuer}] {len(pdfs)} PDF")

        for pdf_file in pdfs:
            pdf_path = os.path.join(issuer_path, pdf_file)
            # Tahun dan tanggal penerbitan dari nama file: YYYYMMDD_...
            date_str = pdf_file[:8] if pdf_file[:8].isdigit() else ""
            issue_year = date_str[:4] if date_str else ""
            try:
                issue_date = datetime.date(
                    int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
                ).isoformat() if date_str else ""
            except ValueError:
                issue_date = ""

            text = _read_cover(pdf_path)
            if not text.strip():
                print(f"  WARN {pdf_file[:50]}: tidak bisa dibaca (image PDF?)")
                continue

            bond_name = extract_bond_name(text)
            series_list = extract_series(text) or [""]  # [""] → satu baris tanpa seri

            print(f"  {pdf_file[:55]}")
            print(f"    Nama  : {bond_name[:80] or '(tidak terdeteksi)'}")
            print(f"    Seri  : {series_list}")

            for seri in series_list:
                # Gunakan tahun dari filename (lebih andal dari teks PDF)
                idx_row = _match_idx(issuer, seri, issue_year, idx_lookup)

                if idx_row:
                    idx_bond_name = idx_row.get("BondName", bond_name)
                    mature_raw = idx_row.get("MatureDate", "")
                    mature_date = mature_raw[:10] if mature_raw else ""
                    try:
                        aktif = datetime.date.fromisoformat(mature_date) >= today
                    except ValueError:
                        aktif = True
                    source = "IDX + Gold"
                    outstanding = idx_row.get("Outstanding", "")
                    rating = idx_row.get("Rating", "")
                    # IDX-matched: dedup by exact IDX bond name
                    dedup_key = (issuer, idx_bond_name)
                else:
                    # Tidak ditemukan di IDX → kemungkinan sudah jatuh tempo
                    # Nama tampilan: gunakan label singkat agar duplikat terdeteksi
                    seri_label = f" Seri {seri}" if seri else ""
                    idx_bond_name = f"{issuer} GSS {issue_year}{seri_label} (tidak di IDX)"
                    mature_date = ""
                    aktif = False
                    source = "Gold Dataset"
                    outstanding = ""
                    rating = ""
                    # Non-IDX: dedup by (issuer, seri, tahun) — satu baris per issuance
                    dedup_key = (issuer, seri, issue_year)
                if dedup_key in seen:
                    # Perbarui PDFFile dengan yang lebih awal (penerbitan pertama)
                    if issue_date < seen[dedup_key].get("IssueDate", "9999"):
                        seen[dedup_key]["IssueDate"] = issue_date
                        seen[dedup_key]["PDFFile"] = pdf_file
                    status = "OK IDX match (duplikat diabaikan)" if idx_row else "WARN tidak di IDX (duplikat diabaikan)"
                else:
                    row_data = {
                        "IssuerCode"   : issuer,
                        "BondName"     : idx_bond_name,
                        "BondNameGold" : bond_name,
                        "Series"       : seri,
                        "GSSType"      : gss_type_from_title(idx_bond_name) or gss_type_from_title(bond_name) or "",
                        "IssueDate"    : issue_date,
                        "MatureDate"   : mature_date,
                        "Outstanding"  : outstanding,
                        "Rating"       : rating,
                        "Aktif"        : aktif,
                        "Source"       : source,
                        "PDFFile"      : pdf_file,
                    }
                    seen[dedup_key] = row_data
                    status = "OK IDX match" if idx_row else "WARN tidak di IDX (kemungkinan jatuh tempo)"
                print(f"    Seri {seri or '-'} -> {status}")

        print()

    rows_out = list(seen.values())

    # Tulis CSV
    if not rows_out:
        print("Tidak ada data yang diekstrak.")
        return

    # Urutkan: aktif dulu, lalu jatuh tempo; dalam grup urutkan berdasarkan issuer
    rows_out.sort(key=lambda r: (not r["Aktif"], r["IssuerCode"], r["BondName"]))

    fieldnames = list(rows_out[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    n_aktif = sum(1 for r in rows_out if r["Aktif"])
    n_jt = len(rows_out) - n_aktif
    print(f"\nOK {len(rows_out)} instrumen unik -> {OUT_CSV}")
    print(f"   Aktif (IDX+Gold): {n_aktif} | Kemungkinan jatuh tempo (Gold only): {n_jt}")
    if n_jt:
        print("   Detail jatuh tempo:")
        for r in rows_out:
            if not r["Aktif"]:
                print(f"     {r['IssuerCode']:6} Seri {r['Series'] or '-'} | {r['BondName'][:70]}")


if __name__ == "__main__":
    main()
