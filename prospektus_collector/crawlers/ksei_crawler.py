import re
import logging
from datetime import date

import aiohttp

from .base_crawler import BaseCrawler, is_pdf_link, extract_date_from_text
from ..config import POJK_18_DATE

logger = logging.getLogger(__name__)

KSEI_DISCLOSURE = "https://www.ksei.co.id/services/disclosure-information"


def is_ksei_prospektus(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in ["prospektus", "penawaran umum", "obligasi", "sukuk"])


class KSEICrawler(BaseCrawler):
    """Crawler untuk KSEI (Kustodian Sentral Efek Indonesia) — keterbukaan informasi."""

    async def run(self, session: aiohttp.ClientSession) -> list:
        logger.info("KSEI: Starting crawl...")
        found_pdfs = []

        all_urls = self.map_site(KSEI_DISCLOSURE)
        pdf_urls = [u for u in all_urls if is_pdf_link(u)]

        for url in pdf_urls:
            filename_part = url.split("/")[-1].split("?")[0]
            if not is_ksei_prospektus(url) and not is_ksei_prospektus(filename_part):
                continue

            doc_date = extract_date_from_text(url) or date.today().isoformat()
            if doc_date < POJK_18_DATE:
                continue

            emiten_code = self._extract_emiten_code(url)
            date_compact = doc_date.replace("-", "")
            filename = f"{date_compact}_{emiten_code}_{filename_part}"

            found_pdfs.append({
                "url": url,
                "filename": filename,
                "emiten_code": emiten_code,
                "source": "ksei",
                "date": doc_date,
            })

        logger.info(f"KSEI: {len(found_pdfs)} PDF ready for download")
        return found_pdfs

    def _extract_emiten_code(self, url: str) -> str:
        m = re.search(r"[_/]([A-Z]{2,6})[_/\.]", url.upper())
        return m.group(1) if m else "UNKNOWN"
