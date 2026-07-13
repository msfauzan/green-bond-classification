#!/usr/bin/env python3
"""
Scraper data Obligasi & Sukuk Korporasi dari website IDX (Bursa Efek Indonesia).

Sumber : https://www.idx.co.id/id/data-pasar/obligasi-sukuk/obligasi-sukuk-korporasi/

Website IDX dilindungi Cloudflare (bot detection) + TLS fingerprinting, jadi
requests/curl biasa pasti diblok (HTTP 403/000). Solusinya: pakai browser Chrome
sungguhan lewat Selenium / undetected-chromedriver supaya lolos challenge, lalu:

  1. Buka halaman -> tunggu Cloudflare lolos & data ter-load.
  2. AUTO-DISCOVERY: baca network log browser untuk menemukan endpoint API JSON
     yang dipanggil halaman (tidak perlu nebak URL-nya).
  3. Tarik semua halaman data via fetch() di dalam konteks browser (mewarisi
     cookie clearance Cloudflare + TLS yang benar), lalu paginate sampai habis.
  4. Simpan ke CSV + JSON di folder data/.
  5. (opsional) Filter hanya emiten pada list_emisi_gss.txt.

Cara pakai:
    pip install -r requirements.txt
    python scrape_idx_bonds.py                 # tarik semua, simpan CSV+JSON
    python scrape_idx_bonds.py --headless       # tanpa jendela (kadang kena CF)
    python scrape_idx_bonds.py --filter ../list_emisi_gss.txt   # filter emiten
    python scrape_idx_bonds.py --debug          # simpan dump mentah utk inspeksi

Catatan: jalankan dengan jendela terlihat (default, non-headless) untuk peluang
terbaik lolos Cloudflare. Kalau gagal, ulangi (challenge kadang butuh 2x).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

# --- shim: undetected-chromedriver butuh distutils (dihapus di Python 3.12+) --- #
try:  # pragma: no cover
    import distutils.version  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    import types

    _dv = types.ModuleType("distutils.version")

    class LooseVersion:  # minimal pengganti distutils.version.LooseVersion
        def __init__(self, vstring=""):
            self.vstring = str(vstring)
            self.version = [int(p) if p.isdigit() else p
                            for p in re.split(r"[._\-]+", self.vstring) if p]

        def _key(self):
            return [p for p in self.version if isinstance(p, int)]

        def __eq__(self, o): return self._key() == LooseVersion(str(o))._key()
        def __lt__(self, o): return self._key() < LooseVersion(str(o))._key()
        def __le__(self, o): return self._key() <= LooseVersion(str(o))._key()
        def __gt__(self, o): return self._key() > LooseVersion(str(o))._key()
        def __ge__(self, o): return self._key() >= LooseVersion(str(o))._key()
        def __str__(self): return self.vstring

    _dv.LooseVersion = LooseVersion
    _dist = types.ModuleType("distutils")
    _dist.version = _dv
    sys.modules.setdefault("distutils", _dist)
    sys.modules["distutils.version"] = _dv
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit, urlencode, parse_qs

PAGE_URL = "https://www.idx.co.id/id/data-pasar/obligasi-sukuk/obligasi-sukuk-korporasi/"

# Endpoint API yang dipakai halaman (hasil reverse-engineer; bondType=1 = korporasi).
# Dipakai sbg cadangan kalau auto-discovery dari network log gagal.
KNOWN_API = ("https://www.idx.co.id/secondary/get/BondSukuk/bond"
             "?pageSize=10&indexFrom=1&bondType=1")

# Kata kunci untuk mengenali endpoint API obligasi pada network log.
API_HINTS = ("bond", "obligasi", "sukuk", "primary", "secondary")

# Nama field yang umum dipakai IDX untuk paginasi & total record.
PAGE_PARAMS = ("start", "indexFrom", "pageNumber", "page", "pageIndex", "draw")
SIZE_PARAMS = ("length", "pageSize", "rows", "limit")
# Param paginasi berbasis OFFSET record (mulai dari nomor record), bukan nomor halaman.
OFFSET_PARAMS = ("start",)
TOTAL_KEYS = ("ResultCount", "resultCount", "recordsTotal", "RecordsTotal",
              "TotalData", "totalData", "total", "Total", "TotalRows",
              "recordsFiltered")
# Berapa record per request saat paginasi-fallback.
DEFAULT_PAGE_SIZE = 100
# Ukuran "ambil semua sekaligus": dipakai kalau total record tidak terdeteksi.
BIG_PAGE_SIZE = 100000

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_driver(headless: bool, prefer_uc: bool = False):
    """Bangun Chrome driver.

    Default pakai Selenium standar (Selenium Manager otomatis cocokkan
    chromedriver dgn versi Chrome, dan biasanya sudah lolos Cloudflare).
    Pakai --uc untuk mencoba undetected-chromedriver dulu kalau Cloudflare
    makin ketat (perlu versi chromedriver yang cocok)."""
    perf_caps = {"performance": "ALL"}

    # --- 1) undetected-chromedriver (opsional, via --uc) ----------------- #
    if prefer_uc:
        try:
            import undetected_chromedriver as uc

            opts = uc.ChromeOptions()
            if headless:
                opts.add_argument("--headless=new")
            opts.add_argument("--start-maximized")
            opts.add_argument("--disable-blink-features=AutomationControlled")
            opts.add_argument("--lang=id-ID")
            opts.set_capability("goog:loggingPrefs", perf_caps)
            driver = uc.Chrome(options=opts)
            print("[i] Driver: undetected-chromedriver")
            return driver
        except Exception as exc:  # noqa: BLE001
            print(f"[!] undetected-chromedriver gagal ({exc}); fallback Selenium standar.")

    # --- 2) Selenium standar --------------------------------------------- #
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--lang=id-ID")
    opts.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.set_capability("goog:loggingPrefs", perf_caps)
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    print("[i] Driver: selenium standar")
    return driver


# --------------------------------------------------------------------------- #
# Cloudflare wait + endpoint discovery
# --------------------------------------------------------------------------- #
def wait_pass_cloudflare(driver, timeout: int = 90) -> None:
    """Tunggu sampai halaman lolos challenge Cloudflare."""
    deadline = time.time() + timeout
    challenge = ("just a moment", "sebentar", "checking your browser",
                 "attention required", "verifying you are human")
    while time.time() < deadline:
        title = (driver.title or "").lower()
        ready = driver.execute_script("return document.readyState")
        if ready == "complete" and not any(c in title for c in challenge):
            # beri waktu XHR data jalan
            time.sleep(4)
            return
        time.sleep(2)
    print("[!] Timeout menunggu Cloudflare (lanjut tetap, mungkin gagal).")


def discover_api_urls(driver) -> list[str]:
    """Ambil URL respons ber-JSON dari network log yang mirip endpoint obligasi."""
    found: dict[str, int] = {}
    try:
        logs = driver.get_log("performance")
    except Exception as exc:  # noqa: BLE001
        print(f"[!] Tidak bisa baca performance log: {exc}")
        return []
    for entry in logs:
        try:
            msg = json.loads(entry["message"])["message"]
        except Exception:  # noqa: BLE001
            continue
        if msg.get("method") != "Network.responseReceived":
            continue
        resp = msg.get("params", {}).get("response", {})
        url = resp.get("url", "")
        mime = (resp.get("mimeType") or "").lower()
        if not url:
            continue
        looks_json = "json" in mime or url.lower().endswith(".json")
        if looks_json and any(h in url.lower() for h in API_HINTS):
            found[url] = found.get(url, 0) + 1
    # urutkan: yang mengandung 'bond' duluan, lalu yang paling sering muncul
    ranked = sorted(found, key=lambda u: ("bond" not in u.lower(), -found[u]))
    return ranked


# --------------------------------------------------------------------------- #
# In-page fetch (mewarisi cookie Cloudflare)
# --------------------------------------------------------------------------- #
FETCH_JS = """
const url = arguments[0];
const done = arguments[arguments.length - 1];
fetch(url, {headers: {'Accept': 'application/json', 'X-Requested-With': 'XMLHttpRequest'},
            credentials: 'include'})
  .then(r => r.text())
  .then(t => done({ok: true, body: t}))
  .catch(e => done({ok: false, error: String(e)}));
"""


def fetch_json(driver, url: str):
    driver.set_script_timeout(60)
    res = driver.execute_async_script(FETCH_JS, url)
    if not res or not res.get("ok"):
        raise RuntimeError(f"fetch gagal: {res}")
    return json.loads(res["body"])


# --------------------------------------------------------------------------- #
# JSON helpers: temukan list record + total
# --------------------------------------------------------------------------- #
def extract_records(payload):
    """Cari list-of-dict terbesar di dalam payload JSON (= array data)."""
    if isinstance(payload, list):
        return payload
    best = []
    if isinstance(payload, dict):
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                if len(v) > len(best):
                    best = v
            elif isinstance(v, (dict, list)):
                cand = extract_records(v)
                if len(cand) > len(best):
                    best = cand
    return best


def extract_total(payload):
    if isinstance(payload, dict):
        for k in TOTAL_KEYS:
            if k in payload and isinstance(payload[k], int):
                return payload[k]
        for v in payload.values():
            if isinstance(v, dict):
                t = extract_total(v)
                if t is not None:
                    return t
    return None


def _dedup_key(r: dict) -> str:
    return json.dumps(r, sort_keys=True, default=str)


def paginate(driver, sample_url: str):
    """Tarik SEMUA record dari endpoint.

    PENTING: API IDX mengembalikan baris dalam urutan yang TIDAK stabil, sehingga
    paginasi per-halaman bisa melewatkan / menduplikasi record (mis. emiten ARKO
    hilang). Karena itu strategi utama: ambil semua dalam SATU request memakai
    pageSize sangat besar. Paginasi+dedup hanya jadi cadangan kalau server membatasi
    pageSize.
    """
    parts = urlsplit(sample_url)
    base = f"{parts.scheme}://{parts.netloc}{parts.path}"
    qs = {k: v[0] for k, v in parse_qs(parts.query).items()}

    page_key = next((k for k in qs if k.lower() in
                     [p.lower() for p in PAGE_PARAMS]), None)
    size_key = next((k for k in qs if k.lower() in
                     [p.lower() for p in SIZE_PARAMS]), None)

    # --- Strategi 1: satu request, ambil semua sekaligus ----------------- #
    if size_key:
        # cari tahu total dulu (request kecil), lalu minta sebanyak itu sekaligus
        probe = dict(qs)
        probe[size_key] = "1"
        if page_key:
            probe[page_key] = "1"
        total = extract_total(fetch_json(driver, f"{base}?{urlencode(probe)}"))
        target = (total + 100) if total else BIG_PAGE_SIZE
        one = dict(qs)
        one[size_key] = str(target)
        if page_key:
            one[page_key] = "1"
        payload = fetch_json(driver, f"{base}?{urlencode(one)}")
        if total is None:
            total = extract_total(payload)
        rows = extract_records(payload)
        if total is not None:
            print(f"[i] Total record menurut server: {total}")
        if rows and (total is None or len(rows) >= total):
            # buang duplikat utk jaga-jaga
            seen, uniq = set(), []
            for r in rows:
                k = _dedup_key(r)
                if k not in seen:
                    seen.add(k)
                    uniq.append(r)
            print(f"[i] Ambil sekaligus dalam 1 request: {len(uniq)} record unik")
            return uniq
        print(f"[!] Single-shot hanya dapat {len(rows)} (server batasi pageSize); "
              "lanjut paginasi + dedup.")

    # --- Strategi 2 (cadangan): paginasi per-halaman + dedup ------------- #
    size = DEFAULT_PAGE_SIZE
    if size_key:
        qs[size_key] = str(size)
    offset_style = bool(page_key) and page_key.lower() in \
        [p.lower() for p in OFFSET_PARAMS]

    all_rows, seen = [], set()
    page, total, stale = 0, None, 0
    while True:
        if page_key:
            qs[page_key] = str(page * size if offset_style else page + 1)
        url = f"{base}?{urlencode(qs)}" if qs else base
        payload = fetch_json(driver, url)
        if total is None:
            total = extract_total(payload)
            if total is not None:
                print(f"[i] Total record menurut server: {total}")
        rows = extract_records(payload)
        if not rows:
            break
        new = 0
        for r in rows:
            key = _dedup_key(r)
            if key not in seen:
                seen.add(key)
                all_rows.append(r)
                new += 1
        print(f"[i] Halaman {page + 1}: +{new} record (total unik {len(all_rows)})")
        if not page_key:                      # endpoint tak ber-paginasi
            break
        if total is not None and len(all_rows) >= total:
            break
        # karena urutan tak stabil, jangan berhenti pada 1 halaman kosong baru;
        # berhenti setelah beberapa halaman beruntun tanpa record baru
        stale = stale + 1 if new == 0 else 0
        if stale >= 8:
            print("[!] Berhenti: beberapa halaman beruntun tanpa record baru.")
            break
        if total is None and len(rows) < size:
            break
        page += 1
        if page > 5000:                       # safety
            break
    return all_rows


# --------------------------------------------------------------------------- #
# HTML table fallback
# --------------------------------------------------------------------------- #
def scrape_html_table(driver):
    """Fallback: baca tabel yang ter-render di halaman (kalau API tak ketemu)."""
    import pandas as pd
    try:
        tables = pd.read_html(driver.page_source)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] Gagal baca tabel HTML: {exc}")
        return []
    if not tables:
        return []
    biggest = max(tables, key=len)
    return biggest.to_dict("records")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def guess_code_field(row: dict) -> str | None:
    for k in row:
        kl = k.lower()
        if kl in ("bondstockid", "stockid", "issuercode", "kodeemiten",
                  "code", "kode", "emiten", "issuer"):
            return k
    return None


def filter_by_emiten(rows: list[dict], codes: set[str]) -> list[dict]:
    if not rows:
        return rows
    field = guess_code_field(rows[0])
    if not field:
        # cari di semua field secara substring
        out = []
        for r in rows:
            blob = " ".join(str(v) for v in r.values()).upper()
            if any(c in blob for c in codes):
                out.append(r)
        return out
    return [r for r in rows if str(r.get(field, "")).upper()[:4] in codes
            or str(r.get(field, "")).upper() in codes]


def clean_rows(rows: list[dict]) -> list[dict]:
    """Buang padding spasi pada nilai string (IDX mengembalikan teks ber-padding)."""
    out = []
    for r in rows:
        out.append({k: (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    return out


def save(rows: list[dict], tag: str = ""):
    import pandas as pd
    rows = clean_rows(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{tag}" if tag else ""
    base = OUT_DIR / f"idx_obligasi_sukuk_korporasi{suffix}_{stamp}"
    df = pd.DataFrame(rows)
    df.to_csv(base.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    base.with_suffix(".json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[OK] {len(rows)} baris -> {base.with_suffix('.csv').name} (+ .json)")
    return df


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Scrape obligasi/sukuk korporasi IDX")
    ap.add_argument("--headless", action="store_true", help="jalankan tanpa jendela")
    ap.add_argument("--filter", metavar="FILE", help="file daftar kode emiten (1/baris)")
    ap.add_argument("--debug", action="store_true", help="dump payload mentah")
    ap.add_argument("--url", default=None, help="paksa pakai endpoint API tertentu")
    ap.add_argument("--uc", action="store_true",
                    help="coba undetected-chromedriver dulu (kalau Cloudflare ketat)")
    args = ap.parse_args()

    codes = None
    if args.filter:
        p = Path(args.filter)
        if not p.exists():
            print(f"[!] File filter tidak ada: {p}")
            return 2
        codes = {ln.strip().upper() for ln in p.read_text().splitlines() if ln.strip()}
        print(f"[i] Filter {len(codes)} emiten: {sorted(codes)}")

    driver = build_driver(args.headless, prefer_uc=args.uc)
    try:
        print(f"[i] Buka {PAGE_URL}")
        driver.get(PAGE_URL)
        wait_pass_cloudflare(driver)

        rows: list[dict] = []
        api_url = args.url
        if not api_url:
            candidates = discover_api_urls(driver)
            if candidates:
                print("[i] Endpoint API terdeteksi:")
                for c in candidates[:8]:
                    print(f"      {c}")
                api_url = candidates[0]
            else:
                print("[!] Endpoint API tak ada di network log; pakai endpoint bawaan.")
                api_url = KNOWN_API

        if api_url:
            print(f"[i] Pakai endpoint: {api_url}")
            try:
                if args.debug:
                    raw = fetch_json(driver, api_url)
                    (OUT_DIR / "_idx_api_sample.json").parent.mkdir(parents=True, exist_ok=True)
                    (OUT_DIR / "_idx_api_sample.json").write_text(
                        json.dumps(raw, indent=2, ensure_ascii=False)[:200000],
                        encoding="utf-8")
                    print("[i] Sample payload disimpan -> data/_idx_api_sample.json")
                rows = paginate(driver, api_url)
            except Exception as exc:  # noqa: BLE001
                print(f"[!] Gagal tarik via API ({exc}); coba fallback tabel HTML.")

        if not rows:
            print("[i] Fallback: scraping tabel HTML yang ter-render.")
            rows = scrape_html_table(driver)

        if not rows:
            print("[X] Tidak ada data berhasil diambil. "
                  "Coba ulangi tanpa --headless, atau jalankan dengan --debug.")
            return 1

        rows = clean_rows(rows)
        if rows:
            print(f"[i] Contoh field: {list(rows[0].keys())}")
        save(rows)
        if codes:
            filtered = filter_by_emiten(rows, codes)
            print(f"[i] Setelah filter emiten: {len(filtered)} baris")
            if filtered:
                save(filtered, tag="gss")
        return 0
    finally:
        try:
            driver.quit()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
