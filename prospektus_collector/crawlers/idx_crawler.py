import re
import logging
from datetime import date
from typing import Optional

import aiohttp

from .base_crawler import BaseCrawler, is_pdf_link, extract_date_from_text
from ..config import PDF_KEYWORDS, POJK_18_DATE

logger = logging.getLogger(__name__)

IDX_BASE = "https://www.idx.co.id"
IDX_EMITEN_LIST = f"{IDX_BASE}/id/perusahaan-tercatat/profil-perusahaan-tercatat"
IDX_ANNOUNCEMENT = f"{IDX_BASE}/id/berita/pengumuman"

# URL path format: /profil-perusahaan-tercatat/KODE  (no query param)
_EMITEN_PATH_PATTERN = re.compile(
    r"/profil-perusahaan-tercatat/([A-Z]{1,6})/?$", re.IGNORECASE
)
# Markdown link: [link_text](url) — capture text and url
_MD_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^\)]+)\)")


def filter_pdf_links_by_keywords(links: list, keywords: list) -> list:
    """
    Filter list URL, return hanya link PDF yang mengandung setidaknya
    satu keyword (case-insensitive) dalam URL-nya.
    """
    result = []
    for link in links:
        lower = link.lower()
        if ".pdf" not in lower:
            continue
        if any(kw.lower().replace(" ", "_") in lower or kw.lower() in lower for kw in keywords):
            result.append(link)
    return result


def extract_prospektus_links_from_markdown(markdown: str, keywords: list) -> list:
    """
    Parse markdown untuk menemukan pasangan (link_text, url) di mana
    link_text mengandung keyword prospektus DAN url mengarah ke PDF.
    IDX menggunakan URL hash, jadi keyword harus dicek di teks link.
    Return: list of URL strings.
    """
    found = []
    for text, url in _MD_LINK_PATTERN.findall(markdown):
        if not is_pdf_link(url):
            continue
        text_lower = text.lower().replace("_", " ").replace("-", " ")
        if any(kw.lower() in text_lower for kw in keywords):
            found.append((text, url))
    return found


class IDXCrawler(BaseCrawler):
    """
    Crawler untuk IDX (Bursa Efek Indonesia).
    - discover_emiten(): return list {"kode": ..., "website": ...}
    - run(): return list dict PDF yang ditemukan di pengumuman IDX
    """

    def discover_emiten(self) -> list:
        """
        Crawl halaman profil emiten IDX untuk mendapatkan daftar emiten
        dan URL website mereka.
        Return: [{"kode": "ABCD", "website": "https://..."}]
        """
        logger.info("IDX: Starting emiten discovery via Firecrawl /map...")
        # Tambahkan limit tinggi untuk mendapatkan lebih banyak emiten
        all_urls = self._map_emiten_with_limit()
        logger.info(f"IDX: Found {len(all_urls)} URLs from map")

        emiten_list = []
        seen_codes = set()

        for url in all_urls:
            m = _EMITEN_PATH_PATTERN.search(url)
            if not m:
                continue
            kode = m.group(1).upper()
            if kode in seen_codes:
                continue
            seen_codes.add(kode)

            page = self.scrape_page(url)
            website = self._extract_website_from_profile(page, url)
            emiten_list.append({"kode": kode, "website": website})
            logger.debug(f"IDX: Discovered emiten {kode} -> {website}")

        logger.info(f"IDX: Discovered {len(emiten_list)} emiten")
        return emiten_list

    def _map_emiten_with_limit(self) -> list:
        """
        Map IDX emiten list dengan limit tinggi.
        Firecrawl default limit = 100; boost ke 5000.
        """
        import time
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self.fc.map(IDX_EMITEN_LIST, limit=5000)
                if hasattr(result, "links") and result.links:
                    return [
                        item.url if hasattr(item, "url") else str(item)
                        for item in result.links
                    ]
                return []
            except Exception as e:
                wait = self.RETRY_BACKOFF ** attempt
                logger.warning(f"IDX map attempt {attempt + 1} failed: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait)
        return []

    def _extract_website_from_profile(self, page: dict, profile_url: str) -> Optional[str]:
        """Ekstrak URL website dari halaman profil emiten IDX."""
        links = page.get("links", [])
        markdown = page.get("markdown", "")

        # Cari external link (bukan idx.co.id)
        for link in links:
            if isinstance(link, str) and "idx.co.id" not in link and link.startswith("http"):
                return link

        # Fallback: cari URL di markdown yang bukan idx
        url_pattern = re.compile(r"https?://(?!.*idx\.co\.id)[^\s\)\"\'\,\]]+")
        m = url_pattern.search(markdown)
        if m:
            return m.group(0).rstrip(".")

        return None

    async def run(self, session: aiohttp.ClientSession) -> list:
        """
        Scrape halaman pengumuman IDX, temukan link PDF prospektus.
        Menggunakan scrape() bukan map() karena IDX adalah JS-rendered page
        dan PDF URLs menggunakan hash (keyword harus dicek di link text).
        Return: [{"url": ..., "filename": ..., "emiten_code": ..., "source": ..., "date": ...}]
        """
        logger.info("IDX: Starting announcement crawl...")
        found_pdfs = []

        # Scrape halaman pengumuman (JS-rendered), dapat markdown + links
        page = self.scrape_page(IDX_ANNOUNCEMENT)
        markdown = page.get("markdown", "")
        all_links = page.get("links", [])

        # Strategi 1: parse markdown untuk link text yang mengandung keyword
        kw_matched = extract_prospektus_links_from_markdown(markdown, PDF_KEYWORDS)
        for link_text, url in kw_matched:
            emiten_code = self._extract_emiten_code_from_text(link_text)
            doc_date = extract_date_from_text(link_text) or extract_date_from_text(markdown[:500]) or date.today().isoformat()
            if doc_date < POJK_18_DATE:
                continue
            filename = self._build_filename(doc_date, emiten_code, link_text)
            found_pdfs.append({
                "url": url,
                "filename": filename,
                "emiten_code": emiten_code,
                "source": "idx",
                "date": doc_date,
            })

        # Strategi 2: fallback — filter PDF links berdasarkan URL keyword
        # (untuk halaman yang URL-nya tidak di-hash)
        plain_pdfs = filter_pdf_links_by_keywords(all_links, PDF_KEYWORDS)
        existing_urls = {p["url"] for p in found_pdfs}
        for url in plain_pdfs:
            if url in existing_urls:
                continue
            emiten_code = self._extract_emiten_code_from_url(url)
            doc_date = extract_date_from_text(url) or date.today().isoformat()
            if doc_date < POJK_18_DATE:
                continue
            filename = self._build_filename(doc_date, emiten_code, url)
            found_pdfs.append({
                "url": url,
                "filename": filename,
                "emiten_code": emiten_code,
                "source": "idx",
                "date": doc_date,
            })

        logger.info(f"IDX: {len(found_pdfs)} PDF ready for download")
        return found_pdfs

    def _extract_emiten_code_from_text(self, text: str) -> str:
        """Ekstrak kode emiten dari link text, mis. 'ABCD' dari '[ABCD]' atau 'ABCD_'."""
        m = re.search(r"\[([A-Z]{2,6})\s*\]", text.upper())
        if m:
            return m.group(1)
        m = re.search(r"[_/]([A-Z]{2,6})[_/\.]", text.upper())
        return m.group(1) if m else "UNKNOWN"

    def _extract_emiten_code_from_url(self, url: str) -> str:
        """Ekstrak kode emiten dari URL jika memungkinkan."""
        m = re.search(r"[_/\[]([A-Z]{2,6})[_/\.\]]", url.upper())
        return m.group(1) if m else "UNKNOWN"

    def _build_filename(self, doc_date: str, emiten_code: str, source_text: str) -> str:
        """Build nama file: YYYYMMDD_KODE_originalname.pdf"""
        date_compact = doc_date.replace("-", "")
        # Ambil bagian akhir dari URL atau teks, bersihkan
        original = source_text.split("/")[-1].split("?")[0]
        if not original.lower().endswith(".pdf"):
            original = re.sub(r"[^\w\-.]", "_", original)[:80] + ".pdf"
        return f"{date_compact}_{emiten_code}_{original}"
