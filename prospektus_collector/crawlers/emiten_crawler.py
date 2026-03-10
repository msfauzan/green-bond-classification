import asyncio
import logging
from datetime import date
from typing import Optional

import aiohttp

from .base_crawler import BaseCrawler, is_pdf_link, extract_date_from_text
from ..config import PDF_KEYWORDS, POJK_18_DATE, MAX_CONCURRENT_EMITEN, EMITEN_IR_KEYWORDS

logger = logging.getLogger(__name__)


def find_ir_url(urls: list) -> Optional[str]:
    """
    Dari list URL satu emiten, temukan URL halaman Investor Relations.
    Return URL pertama yang cocok, atau None.
    """
    for url in urls:
        lower = url.lower()
        if any(kw in lower for kw in EMITEN_IR_KEYWORDS):
            return url
    return None


class EmitenCrawler(BaseCrawler):
    """
    Crawler untuk website emiten.
    Terima list emiten (kode + website), temukan halaman IR, ekstrak PDF.
    """

    async def run(self, emiten_list: list, session: aiohttp.ClientSession) -> list:
        """
        Crawl semua emiten secara concurrent (dibatasi MAX_CONCURRENT_EMITEN).
        Return flat list semua PDF yang ditemukan.
        """
        valid_emiten = [e for e in emiten_list if e.get("website")]
        logger.info(f"Emiten: Starting crawl for {len(valid_emiten)} emiten...")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMITEN)
        tasks = [
            self._crawl_one_emiten(emiten, session, semaphore)
            for emiten in valid_emiten
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        found_pdfs = []
        for r in results:
            if isinstance(r, list):
                found_pdfs.extend(r)
            elif isinstance(r, Exception):
                logger.warning(f"Emiten crawl error: {r}")

        logger.info(f"Emiten: {len(found_pdfs)} PDF found across all emiten")
        return found_pdfs

    async def _crawl_one_emiten(
        self,
        emiten: dict,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> list:
        async with semaphore:
            kode = emiten["kode"]
            website = emiten["website"]
            found = []

            try:
                all_urls = self.map_site(website)
                ir_url = find_ir_url(all_urls)

                if not ir_url:
                    logger.debug(f"Emiten {kode}: No IR page found at {website}")
                    return []

                page = self.scrape_page(ir_url)
                links = page.get("links", [])
                pdf_links = [l for l in links if is_pdf_link(l)]

                for url in pdf_links:
                    lower = url.lower()
                    if not any(kw in lower for kw in PDF_KEYWORDS):
                        continue

                    doc_date = extract_date_from_text(url) or date.today().isoformat()
                    if doc_date < POJK_18_DATE:
                        continue

                    date_compact = doc_date.replace("-", "")
                    original = url.split("/")[-1].split("?")[0]
                    filename = f"{date_compact}_{kode}_{original}"

                    found.append({
                        "url": url,
                        "filename": filename,
                        "emiten_code": kode,
                        "source": "emiten",
                        "date": doc_date,
                    })

            except Exception as e:
                logger.warning(f"Emiten {kode}: Error crawling {website}: {e}")

            return found
