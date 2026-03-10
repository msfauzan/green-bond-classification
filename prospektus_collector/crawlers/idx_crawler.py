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

_EMITEN_PROFILE_PATTERN = re.compile(r"kodeEmiten=([A-Z]{1,6})", re.IGNORECASE)


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
        all_urls = self.map_site(IDX_EMITEN_LIST)
        logger.info(f"IDX: Found {len(all_urls)} URLs from map")

        emiten_list = []
        seen_codes = set()

        for url in all_urls:
            m = _EMITEN_PROFILE_PATTERN.search(url)
            if not m:
                continue
            kode = m.group(1).upper()
            if kode in seen_codes:
                continue
            seen_codes.add(kode)

            page = self.scrape_page(url)
            website = self._extract_website_from_profile(page)
            emiten_list.append({"kode": kode, "website": website})
            logger.debug(f"IDX: Discovered emiten {kode} -> {website}")

        logger.info(f"IDX: Discovered {len(emiten_list)} emiten")
        return emiten_list

    def _extract_website_from_profile(self, page: dict) -> Optional[str]:
        """Ekstrak URL website dari halaman profil emiten IDX."""
        links = page.get("links", [])
        markdown = page.get("markdown", "")

        for link in links:
            if "idx.co.id" not in link and link.startswith("http"):
                return link

        url_pattern = re.compile(r"https?://(?!.*idx\.co\.id)[^\s\)\"\'\,]+")
        m = url_pattern.search(markdown)
        if m:
            return m.group(0)

        return None

    async def run(self, session: aiohttp.ClientSession) -> list:
        """
        Scrape halaman pengumuman IDX, temukan link PDF prospektus.
        Return: [{"url": ..., "filename": ..., "emiten_code": ..., "source": ..., "date": ...}]
        """
        logger.info("IDX: Starting announcement crawl...")
        found_pdfs = []

        announcement_urls = self.map_site(IDX_ANNOUNCEMENT)
        pdf_urls = [u for u in announcement_urls if is_pdf_link(u)]
        filtered = filter_pdf_links_by_keywords(pdf_urls, PDF_KEYWORDS)
        logger.info(f"IDX: Found {len(filtered)} PDF links after keyword filter")

        for url in filtered:
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

    def _extract_emiten_code_from_url(self, url: str) -> str:
        """Ekstrak kode emiten dari URL jika memungkinkan."""
        m = re.search(r"[_/\[]([A-Z]{2,6})[_/\.\]]", url.upper())
        return m.group(1) if m else "UNKNOWN"

    def _build_filename(self, doc_date: str, emiten_code: str, url: str) -> str:
        """Build nama file: YYYYMMDD_KODE_originalname.pdf"""
        date_compact = doc_date.replace("-", "")
        original = url.split("/")[-1].split("?")[0]
        return f"{date_compact}_{emiten_code}_{original}"
