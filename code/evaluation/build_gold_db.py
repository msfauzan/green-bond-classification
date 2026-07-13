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
sys.path.insert(0, os.path.join(ROOT, "code"))

from classifier.engine import extract_series
from classifier.title_lookup import (
    gss_type_from_title, _load_idx, TITLE_GSS_MARKERS, _is_gss_titled, IDX_CSV,
)

GOLD_GSS = os.path.join(ROOT, "data", "pdf_by_content", "01_prospektus_utama", "0. Fix GSS", "GSS")
OUT_CSV  = os.path.join(ROOT, "data", "gold_bonds_db.csv")
LP = "\\\\?\\"


# ---------------------------------------------------------------------------
# Baca PDF (long-path safe, hanya 5 halaman sampul)
# ---------------------------------------------------------------------------

def _read_cover(pdf_path: str, n_pages: int = 5) -> str:
    import fitz
    text = ""
    for p in (LP + pdf_path, pdf_path):
        try:
            doc = fitz.open(p)
            try:
                text = "\n".join(doc[i].get_text() for i in range(min(n_pages, len(doc))))
                break
            finally:
                doc.close()
        except Exception:
            continue
    if text.strip():
        return text
    # PDF hasil scan: pakai sidecar hasil OCR bila ada (ocr_scanned_gss.py)
    for sc in (LP + pdf_path + ".txt", pdf_path + ".txt"):
        try:
            with open(sc, encoding="utf-8") as f:
                return f.read()
        except OSError:
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


_DAY_RE = re.compile(r"^\W*(senin|selasa|rabu|kamis|jumat|sabtu|minggu)\b")
_KORAN = ("investor daily", "kontan", "neraca", "bisnis indonesia",
          "media indonesia")


def _norm_ds(s: str) -> str:
    """Normalisasi + rapikan huruf ber-spasi tipografis (P R O S P E K T U S)."""
    s = re.sub(r"\s+", " ", s.lower()).strip()
    return re.sub(r"\b(?:\w ){3,}\w\b", lambda m: m.group(0).replace(" ", ""), s)


def _read_pages(pdf_path: str) -> list[str]:
    """Teks SEMUA halaman; fallback sidecar OCR (satu 'halaman') untuk scan."""
    import fitz
    for p in (LP + pdf_path, pdf_path):
        try:
            doc = fitz.open(p)
            try:
                pages = [doc[i].get_text() for i in range(len(doc))]
            finally:
                doc.close()
            if any(t.strip() for t in pages):
                return pages
            break
        except Exception:
            continue
    for sc in (LP + pdf_path + ".txt", pdf_path + ".txt"):
        try:
            with open(sc, encoding="utf-8") as f:
                return [f.read()]
        except OSError:
            continue
    return []


def doc_type(pdf_path: str) -> tuple[str, int]:
    """Jenis dokumen + peringkat substansi, dari judul di tingkat HALAMAN
    (seluruh halaman dibaca — bundel pengantar+dokumen tetap terdeteksi).

    Peringkat memilih dokumen terbaik per instrumen: 5 Prospektus penuh,
    4 Informasi Tambahan penuh, 3 ringkas (PR/ITR), 2 iklan koran,
    1 tak teridentifikasi, 0 pengantar/kosong, -1 pemeringkatan.
    """
    pages = _read_pages(pdf_path)
    n = len(pages)
    for i, pt in enumerate(pages):
        h = _norm_ds(pt)[:400]
        if not h:
            continue
        if "hasil pemeringkatan" in h:
            return "Pemeringkatan", -1
        if h.startswith("nomor surat") or "perihal" in h[:120]:
            continue   # halaman pengantar e-form — periksa halaman berikutnya
        sisa = n - i
        if sisa <= 8 and (_DAY_RE.match(h) or h.startswith("jadwal")
                          or any(k in h[:200] for k in _KORAN)):
            return "Iklan Ringkas", 2
        if "informasi tambahan ringkas" in h:
            return "Info Tambahan Ringkas", 3
        if "prospektus ringkas" in h:
            return "Prospektus Ringkas", 3
        if h.startswith("tambahan informasi"):
            return "Info Tambahan Ringkas", 3
        if h.startswith("informasi tambahan"):
            return ("Informasi Tambahan", 4) if sisa >= 40 \
                else ("Info Tambahan Ringkas", 3)
        if h.startswith("prospektus"):
            return ("Prospektus", 5) if sisa >= 40 else ("Prospektus Ringkas", 3)
        body = _norm_ds(" ".join(pages[i:i + 3]))
        if "informasi tambahan ringkas" in body:
            return "Info Tambahan Ringkas", 3
        if "prospektus ringkas" in body:
            return "Prospektus Ringkas", 3
        if sisa >= 60:
            return "Prospektus", 5
        return "Dokumen Emisi", 1
    return "Kosong/Scan", 0


def _n_pages(pdf_path: str) -> int:
    import fitz
    for p in (LP + pdf_path, pdf_path):
        try:
            doc = fitz.open(p)
            try:
                return len(doc)
            finally:
                doc.close()
        except Exception:
            continue
    return 0


def _match_idx_all(issuer: str, cover_text: str, idx_lookup: dict) -> list[dict]:
    """
    Semua baris IDX yang konsisten dengan SAMPUL PDF (satu prospektus bisa
    mencakup beberapa seri, dan bundel obligasi+sukuk sekaligus).

    Baris IDX cocok bila setiap atribut pada namanya juga muncul di sampul:
      (1) minimal satu penanda GSS yang ada di nama IDX muncul di sampul,
      (2) "tahun YYYY" dari nama IDX muncul di sampul,
      (3) "tahap N" dari nama IDX (bila ada) muncul di sampul,
      (4) "seri X" dari nama IDX (bila ada) muncul di sampul,
      (5) kata instrumen ("sukuk"/"obligasi"/"surat berharga perpetual") muncul.
    """
    cover = re.sub(r"\s+", " ", cover_text.lower())
    out: list[dict] = []
    for row in idx_lookup.get(issuer, []):
        idx_name = re.sub(r"\s+", " ", row.get("BondName", "").lower())

        markers = [m for m in TITLE_GSS_MARKERS if m in idx_name]
        if not markers or not any(m in cover for m in markers):
            continue

        tahun = re.search(r"tahun (\d{4})", idx_name)
        if tahun and f"tahun {tahun.group(1)}" not in cover:
            continue

        tahap = re.search(r"tahap ([ivx]+|\d+)\b", idx_name)
        if tahap and f"tahap {tahap.group(1)}" not in cover:
            continue

        seri = re.search(r"\bseri ([a-z])\b", idx_name)
        if seri and f"seri {seri.group(1)}" not in cover:
            continue

        for instr in ("surat berharga perpetual", "sukuk", "obligasi"):
            if idx_name.startswith(instr):
                if instr not in cover:
                    idx_name = None  # instrumen tak disebut di sampul
                break
        if idx_name is None:
            continue

        out.append(row)
    return out


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

            dtype, rank = doc_type(pdf_path)
            if rank < 0:   # pemeringkatan dkk — bukan dokumen emisi
                print(f"  {pdf_file[:55]} -> SKIP ({dtype})")
                continue

            print(f"  {pdf_file[:55]}")
            print(f"    Nama  : {bond_name[:80] or '(tidak terdeteksi)'} [{dtype}]")
            print(f"    Seri  : {series_list}")

            # Satu PDF bisa cocok ke BANYAK baris IDX (multi-seri, bundel
            # obligasi+sukuk); cocokkan atribut nama IDX langsung ke sampul.
            idx_rows = _match_idx_all(issuer, text, idx_lookup)
            # Seri pada dokumen yang tidak ter-cover match IDX = kemungkinan
            # sudah jatuh tempo/delisting -> baris gold-only. Syarat: dokumen
            # substantif (rank>=3) dan seri teridentifikasi — bundel/iklan
            # tanpa seri bukan bukti instrumen (fantom).
            matched_series = set()
            for r in idx_rows:
                m = re.search(r"\bSeri ([A-Z])\b", r.get("BondName", ""))
                matched_series.add(m.group(1) if m else "")
            emits: list[tuple[dict | None, str]] = [(r, "") for r in idx_rows]
            # iklan resmi pun memuat rincian seri — bukti sah. Tapi bila match
            # IDX-nya instrumen tanpa penamaan seri (mis. perpetual), seluruh
            # emisi dianggap ter-cover.
            if rank >= 2 and (not idx_rows or matched_series - {""}):
                emits += [(None, s) for s in series_list
                          if s and s not in matched_series]
            if not emits:
                print("    (tanpa match IDX & tanpa seri — dilewati)")
                continue

            for idx_row, seri in emits:
                if idx_row:
                    idx_bond_name = idx_row.get("BondName", bond_name)
                    m = re.search(r"\bSeri ([A-Z])\b", idx_bond_name)
                    seri = m.group(1) if m else ""
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
                quality = (rank, _n_pages(pdf_path))
                if dedup_key in seen:
                    # Pilih dokumen paling substantif (peringkat jenis, lalu
                    # jumlah halaman) — bukan sekadar yang paling awal.
                    if quality > seen[dedup_key].get("_quality", (-9, 0)):
                        seen[dedup_key].update(
                            IssueDate=issue_date, PDFFile=pdf_file,
                            DocType=dtype, _quality=quality)
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
                        "DocType"      : dtype,
                        "_quality"     : quality,
                    }
                    seen[dedup_key] = row_data
                    status = "OK IDX match" if idx_row else "WARN tidak di IDX (kemungkinan jatuh tempo)"
                print(f"    {idx_bond_name[:60]} -> {status}")

        print()

    rows_out = list(seen.values())
    for r in rows_out:
        r.pop("_quality", None)

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
