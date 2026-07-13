#!/usr/bin/env python3
"""
Scraper prospektus obligasi/sukuk dari IDX Announcement.

Sumber : https://www.idx.co.id/id/berita/pengumuman/
API    : https://www.idx.co.id/primary/NewsAnnouncement/GetAllAnnouncement

Alur:
  1. Buka halaman pengumuman IDX via Selenium (lolos Cloudflare).
  2. Fetch semua halaman dengan keyword 'prospektus'.
  3. Filter item yang kode emiten-nya ada di daftar target.
  4. Unduh setiap lampiran PDF via FullSavePath (URL langsung dari API).
  5. Simpan ke output_dir dengan struktur folder yang konsisten dengan koleksi
     yang sudah ada di pdf_by_content/01_prospektus_utama/.

Cara pakai:
    pip install -r requirements.txt
    python scrape_prospektus.py                         # 15 emiten GSS
    python scrape_prospektus.py --codes BBRI BMRI SMFP  # emiten tertentu
    python scrape_prospektus.py --all                   # semua emiten di IDX
    python scrape_prospektus.py --from-date 2024-01-01  # mulai tanggal tertentu
    python scrape_prospektus.py --headless              # tanpa jendela browser
    python scrape_prospektus.py --debug                 # simpan payload JSON mentah
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

# --------------------------------------------------------------------------- #
# Konstanta
# --------------------------------------------------------------------------- #

PAGE_URL = "https://www.idx.co.id/id/berita/pengumuman/"
API_BASE = "https://www.idx.co.id/primary/NewsAnnouncement/GetAllAnnouncement"

# Kode emiten GSS yang menjadi target default
DEFAULT_GSS_CODES = {
    "ARKO", "BBNI", "BBRI", "BBTN", "BJBR", "BMRI",
    "BRIS", "FIFA", "IIFF", "ISSP", "OPPM", "PNMP",
    "POLI", "PPGD", "SMFP", "SMII",
}

PAGE_SIZE = 50  # maksimum per halaman API
KEYWORD = "prospektus"

OUT_DIR = Path(__file__).resolve().parent.parent / "pdf_by_content" / "01_prospektus_utama"
# Windows path limit 260 chars; base OUT_DIR ~102 chars → sisa ~75 untuk nama folder/file
MAX_NAME = 75


# --------------------------------------------------------------------------- #
# Selenium
# --------------------------------------------------------------------------- #

_DOWNLOAD_TMP = Path(__file__).resolve().parent / "_dl_tmp"


def build_driver(headless: bool):
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options

    _DOWNLOAD_TMP.mkdir(exist_ok=True)

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
    opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})
    # PDF langsung diunduh ke folder, tidak dibuka di viewer Chrome
    opts.add_experimental_option("prefs", {
        "download.default_directory": str(_DOWNLOAD_TMP),
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,
    })

    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"},
    )
    # Pastikan CDP download behavior juga mengarah ke folder yang sama
    driver.execute_cdp_cmd(
        "Page.setDownloadBehavior",
        {"behavior": "allow", "downloadPath": str(_DOWNLOAD_TMP)},
    )
    return driver


def wait_cloudflare(driver, timeout: int = 90) -> None:
    deadline = time.time() + timeout
    challenge = ("just a moment", "sebentar", "checking your browser",
                 "attention required", "verifying you are human")
    while time.time() < deadline:
        title = (driver.title or "").lower()
        if driver.execute_script("return document.readyState") == "complete" \
                and not any(c in title for c in challenge):
            time.sleep(3)
            return
        time.sleep(2)
    print("[!] Timeout Cloudflare — lanjut tetap.")


# --------------------------------------------------------------------------- #
# In-browser fetch (mewarisi cookie Cloudflare)
# --------------------------------------------------------------------------- #

_FETCH_JS = """
const url = arguments[0];
const done = arguments[arguments.length - 1];
fetch(url, {
  headers: {'Accept': 'application/json', 'X-Requested-With': 'XMLHttpRequest'},
  credentials: 'include'
})
  .then(r => r.text())
  .then(t => done({ok: true, body: t}))
  .catch(e => done({ok: false, error: String(e)}));
"""

def fetch_json(driver, url: str):
    driver.set_script_timeout(60)
    res = driver.execute_async_script(_FETCH_JS, url)
    if not res or not res.get("ok"):
        raise RuntimeError(f"fetch gagal: {res}")
    return json.loads(res["body"])


def download_via_chrome(driver, url: str, dest: Path,
                        timeout: int = 90) -> bool:
    """
    Unduh PDF lewat tab baru Chrome (CDP download).
    Menggunakan switch_to.new_window() — tidak kena popup blocker.
    """
    for f in _DOWNLOAD_TMP.iterdir():
        if f.is_file():
            try:
                f.unlink()
            except Exception:
                pass

    main_handle = driver.current_window_handle

    try:
        driver.switch_to.new_window("tab")
        driver.get(url)
    except Exception as e:
        print(f"      [!] Gagal buka tab: {e}")
        try:
            driver.switch_to.window(main_handle)
        except Exception:
            pass
        return False

    deadline = time.time() + timeout
    success = False
    while time.time() < deadline:
        files = [f for f in _DOWNLOAD_TMP.iterdir()
                 if f.is_file() and f.suffix != ".crdownload"]
        crdowns = list(_DOWNLOAD_TMP.glob("*.crdownload"))
        if files and not crdowns:
            try:
                files[0].rename(dest)
                success = True
            except Exception as e:
                print(f"      [!] Gagal pindah file: {e}")
            break
        time.sleep(1)

    try:
        driver.close()
    except Exception:
        pass
    try:
        driver.switch_to.window(main_handle)
    except Exception:
        pass

    return success


# --------------------------------------------------------------------------- #
# Parse respons API
# --------------------------------------------------------------------------- #

def parse_items(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("Items", "items", "Announcements", "Results", "Data"):
            v = payload.get(k)
            if isinstance(v, list):
                return v
    return []


def item_code(item: dict) -> str:
    return (item.get("Code") or "").strip()


def item_date(item: dict) -> str:
    """Kembalikan tanggal dalam format YYYYMMDD."""
    raw = item.get("PublishDate") or item.get("Date") or ""
    digits = re.sub(r"\D", "", raw)
    return digits[:8] if len(digits) >= 8 else datetime.now().strftime("%Y%m%d")


def item_title(item: dict) -> str:
    return (item.get("Title") or item.get("AnnouncementTitle") or "Pengumuman").strip()


def item_id(item: dict) -> str:
    """Ekstrak ID numerik dari JmsxGroupId (e.g. 'f-32101109-0' → '32101109')."""
    raw = item.get("JmsxGroupId") or item.get("Id") or ""
    nums = re.findall(r"\d+", str(raw))
    # Ambil angka terpanjang (bukan 0)
    candidates = [n for n in nums if len(n) >= 5]
    return candidates[0] if candidates else raw


def item_attachments(item: dict) -> list[dict]:
    """Kembalikan list lampiran dari Attachments atau PdfPath."""
    atts = item.get("Attachments")
    if isinstance(atts, list) and atts:
        return atts
    # Fallback: parse JSON string di PdfPath
    pdf_path = item.get("PdfPath") or ""
    if pdf_path:
        try:
            return json.loads(pdf_path)
        except Exception:
            pass
    return []


# --------------------------------------------------------------------------- #
# Nama file & folder (aman Windows)
# --------------------------------------------------------------------------- #

def safe(s: str, maxlen: int = MAX_NAME) -> str:
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    s = re.sub(r"\s+", " ", s).strip(". ")
    # rstrip lagi setelah truncation: nama folder Windows tak boleh
    # berakhir spasi/titik (move/resolve gagal diam-diam)
    return s[:maxlen].rstrip(". ")


def make_folder_name(item: dict) -> str:
    date = item_date(item)
    code = item_code(item)
    title = item_title(item)
    return safe(f"{date}_{code}_{title}")


def make_file_name(att: dict, item: dict, lamp_idx: int) -> str:
    """
    Gunakan OriginalFilename dari API jika ada.
    OriginalFilename format: 'YYYYMMDD_CODE_Judul_ID_lampN.pdf'
    atau 'YYYYMMDD_CODE_Judul/SubJudul_ID_lampN.pdf' (ada slash = sub-path).
    Kita ambil bagian terakhir setelah '/' sebagai nama file.
    """
    orig = att.get("OriginalFilename") or ""
    if orig:
        # Ganti slash jadi underscore (Windows-safe)
        fname = orig.replace("/", "_").replace("\\", "_")
        fname = safe(fname)
        if fname:
            return fname

    # Fallback: konstruksi dari data item
    date = item_date(item)
    code = item_code(item)
    title = item_title(item)
    ann_id = item_id(item)
    return safe(f"{date}_{code}_{title}_{ann_id}_lamp{lamp_idx}.pdf")


# --------------------------------------------------------------------------- #
# Fetch semua pengumuman (semua halaman)
# --------------------------------------------------------------------------- #

def fetch_all_announcements(driver, keyword: str, from_date: str | None,
                            target_codes: set[str] | None) -> list[dict]:
    """
    Paginasi penuh. Jika target_codes diberikan, berhenti lebih awal jika semua
    kode sudah terkumpul (optimasi). Kembalikan list item yang sudah difilter.
    """
    all_items: list[dict] = []
    seen_ids: set[str] = set()
    page = 1

    while True:
        params: dict[str, str | int] = {
            "keywords": keyword,
            "pageNumber": page,
            "pageSize": PAGE_SIZE,
            "lang": "id",
        }
        if from_date:
            params["startDate"] = from_date

        url = f"{API_BASE}?{urlencode(params)}"
        payload = None
        for attempt in range(3):
            try:
                payload = fetch_json(driver, url)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(3)
                else:
                    print(f"  [!] Halaman {page} gagal setelah 3x: {e}")
        if payload is None:
            break

        items = parse_items(payload)
        if not items:
            print(f"  [i] Halaman {page}: kosong — selesai.")
            break

        new = 0
        for item in items:
            iid = item.get("Id") or item.get("JmsxGroupId") or str(id(item))
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
            code = item_code(item)
            if target_codes is None or code in target_codes:
                all_items.append(item)
                new += 1

        total_on_page = len(items)
        matched_so_far = len(all_items)
        print(f"  [i] Halaman {page}: {total_on_page} item, {new} cocok "
              f"(total cocok: {matched_so_far})")

        if total_on_page < PAGE_SIZE:
            print(f"  [i] Halaman terakhir ({total_on_page} < {PAGE_SIZE}) — selesai.")
            break

        page += 1
        time.sleep(0.3)

    return all_items


# --------------------------------------------------------------------------- #
# Download satu item
# --------------------------------------------------------------------------- #

def download_item(driver, item: dict, out_dir: Path, debug: bool) -> int:
    folder_name = make_folder_name(item)
    folder = out_dir / folder_name
    folder.mkdir(parents=True, exist_ok=True)

    if debug:
        (folder / "_payload.json").write_text(
            json.dumps(item, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8")

    attachments = item_attachments(item)
    if not attachments:
        print(f"    [!] Tidak ada lampiran.")
        return 0

    downloaded = 0
    for lamp_idx, att in enumerate(attachments, start=1):
        url = att.get("FullSavePath") or att.get("url") or ""
        if not url:
            continue
        if url.startswith("/"):
            url = "https://www.idx.co.id" + url

        fname = make_file_name(att, item, lamp_idx)
        dest = folder / fname

        if dest.exists() and dest.stat().st_size > 500:
            print(f"    [skip] {fname}")
            downloaded += 1
            continue

        print(f"    [DL] lamp{lamp_idx}: {fname}")
        ok = download_via_chrome(driver, url, dest)
        if ok:
            size_kb = dest.stat().st_size // 1024
            print(f"    [OK] {size_kb:,} KB")
            downloaded += 1
        else:
            print(f"    [!] Gagal / timeout: {url}")

    return downloaded


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Scrape prospektus IDX untuk emiten GSS")
    ap.add_argument("--codes", nargs="+", metavar="CODE",
                    help="Kode emiten target (default: 15 emiten GSS)")
    ap.add_argument("--all", action="store_true",
                    help="Unduh untuk semua emiten (tanpa filter kode)")
    ap.add_argument("--keyword", default=KEYWORD,
                    help=f"Keyword pencarian IDX (default: '{KEYWORD}')")
    ap.add_argument("--from-date", metavar="YYYY-MM-DD",
                    help="Hanya pengumuman mulai tanggal ini")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--debug", action="store_true",
                    help="Simpan payload JSON mentah per folder")
    ap.add_argument("--out-dir", metavar="PATH",
                    help="Override folder output")
    args = ap.parse_args()

    # Target emiten
    if args.all:
        target_codes = None
        print(f"[i] Mode: semua emiten (keyword='{args.keyword}')")
    elif args.codes:
        target_codes = {c.upper() for c in args.codes}
        print(f"[i] Target {len(target_codes)} emiten: {sorted(target_codes)}")
    else:
        target_codes = DEFAULT_GSS_CODES
        print(f"[i] Target {len(target_codes)} emiten GSS: {sorted(target_codes)}")

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[i] Output: {out_dir}")

    driver = build_driver(args.headless)
    try:
        print(f"[i] Buka {PAGE_URL} (tunggu Cloudflare)...")
        driver.get(PAGE_URL)
        wait_cloudflare(driver)
        print("[i] Halaman siap.\n")

        print(f"[i] Fetch semua pengumuman keyword='{args.keyword}'...")
        items = fetch_all_announcements(
            driver, args.keyword, args.from_date, target_codes)

        # Ringkasan per emiten
        from collections import Counter
        code_count = Counter(item_code(i) for i in items)
        print(f"\n[i] {len(items)} pengumuman cocok:")
        for code, n in sorted(code_count.items()):
            print(f"    {code}: {n}")

        total_dl = 0
        for idx, item in enumerate(items, 1):
            code = item_code(item)
            date = item_date(item)
            title = item_title(item)[:70]
            print(f"\n[{idx}/{len(items)}] {date} {code} | {title}")
            try:
                n = download_item(driver, item, out_dir, args.debug)
            except Exception as e:
                print(f"    [!] Error, lanjut: {e}")
                n = 0
            total_dl += n
            time.sleep(0.4)

        print(f"\n[SELESAI] {total_dl} file diunduh dari {len(items)} pengumuman.")
        print(f"[i] Output: {out_dir}")
        return 0

    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
