import re
import logging
from datetime import date

import aiohttp

from .base_crawler import BaseCrawler, is_pdf_link, extract_date_from_text
from ..config import POJK_18_DATE

logger = logging.getLogger(__name__)

OJK_EFEK = "https://www.ojk.go.id/id/kanal/pasar-modal/data-dan-statistik/penawaran-umum"

_DEBT_KEYWORDS = ["obligasi", "sukuk", "eba", "mtn", "medium term", "utang"]


def is_debt_security_prospektus(text: str) -> bool:
    """Return True jika teks mengindikasikan prospektus efek bersifat utang."""
    # Normalise underscores to spaces for URL-style text
    lower = text.lower().replace("_", " ")
    if not any(kw in lower for kw in ["prospektus", "penawaran umum"]):
        return False
    return any(kw in lower for kw in _DEBT_KEYWORDS)


class OJKCrawler(BaseCrawler):
    """Crawler untuk OJK (Otoritas Jasa Keuangan) — seksi publikasi efek."""

    async def run(self, session: aiohttp.ClientSession) -> list:
        logger.info("OJK: Starting crawl...")
        found_pdfs = []

        all_urls = self.map_site(OJK_EFEK)
        pdf_urls = [u for u in all_urls if is_pdf_link(u)]

        for url in pdf_urls:
            filename_part = url.split("/")[-1].split("?")[0]
            if not is_debt_security_prospektus(url) and not is_debt_security_prospektus(filename_part):
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
                "source": "ojk",
                "date": doc_date,
            })

        logger.info(f"OJK: {len(found_pdfs)} PDF ready for download")
        return found_pdfs

    def _extract_emiten_code(self, url: str) -> str:
        m = re.search(r"[_/]([A-Z]{2,6})[_/\.]", url.upper())
        return m.group(1) if m else "UNKNOWN"
