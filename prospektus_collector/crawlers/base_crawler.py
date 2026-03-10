import re
import asyncio
import logging
import time
from typing import Optional

import aiohttp
from firecrawl import FirecrawlApp

logger = logging.getLogger(__name__)

_PDF_PATTERN = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)
_DATE_ISO = re.compile(r"(\d{4}-\d{2}-\d{2})")
_DATE_SLASH = re.compile(r"(\d{2})/(\d{2})/(\d{4})")


def is_pdf_link(url: str) -> bool:
    """Return True jika URL mengarah ke file PDF."""
    return bool(_PDF_PATTERN.search(url))


def extract_date_from_text(text: str) -> Optional[str]:
    """
    Ekstrak tanggal dari teks. Return format ISO YYYY-MM-DD atau None.
    Coba format ISO dulu, lalu DD/MM/YYYY.
    """
    m = _DATE_ISO.search(text)
    if m:
        return m.group(1)
    m = _DATE_SLASH.search(text)
    if m:
        day, month, year = m.group(1), m.group(2), m.group(3)
        return f"{year}-{month}-{day}"
    return None


class BaseCrawler:
    """
    Base class untuk semua crawler.
    Menyediakan: Firecrawl client, async PDF downloader dengan retry.
    """

    MAX_RETRIES = 3
    RETRY_BACKOFF = 2  # detik, dikali 2^attempt

    def __init__(self, firecrawl_api_key: str, timeout: int = 30):
        self.fc = FirecrawlApp(api_key=firecrawl_api_key)
        self.timeout = timeout

    async def download_pdf(
        self,
        url: str,
        session: aiohttp.ClientSession,
        retries: int = MAX_RETRIES,
    ) -> Optional[bytes]:
        """
        Download PDF dari URL. Return bytes atau None jika gagal.
        Retry dengan exponential backoff.
        """
        for attempt in range(retries):
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as resp:
                    if resp.status == 200:
                        return await resp.read()
                    logger.warning(f"HTTP {resp.status} for {url}")
                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                wait = self.RETRY_BACKOFF ** attempt
                logger.warning(
                    f"Download attempt {attempt + 1}/{retries} failed for {url}: {e}. Retry in {wait}s"
                )
                if attempt < retries - 1:
                    await asyncio.sleep(wait)
        logger.error(f"All {retries} download attempts failed for {url}")
        return None

    def scrape_page(self, url: str) -> dict:
        """
        Panggil Firecrawl /scrape. Retry dengan backoff.
        Return dict {"links": [...], "markdown": "..."} atau {} jika gagal.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self.fc.scrape(url, formats=["markdown", "links"])
                raw_links = result.links or []
                # LinkResult objects have .url; plain strings pass through
                links = [
                    item.url if hasattr(item, "url") else str(item)
                    for item in raw_links
                ]
                return {
                    "links": links,
                    "markdown": result.markdown or "",
                }
            except Exception as e:
                wait = self.RETRY_BACKOFF ** attempt
                logger.warning(f"Firecrawl scrape attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    # Synchronous sleep — Firecrawl calls are synchronous
                    time.sleep(wait)
        return {}

    def map_site(self, url: str) -> list:
        """
        Panggil Firecrawl /map untuk mendapat semua URL di domain.
        Return list of URL strings atau [] jika gagal.
        """
        for attempt in range(self.MAX_RETRIES):
            try:
                result = self.fc.map(url)
                # firecrawl-py v4: result is MapData with .links = list[LinkResult]
                if hasattr(result, "links") and result.links:
                    links = result.links
                    # LinkResult objects have .url attribute; plain strings pass through
                    return [
                        item.url if hasattr(item, "url") else str(item)
                        for item in links
                    ]
                if isinstance(result, list):
                    return [
                        item.url if hasattr(item, "url") else str(item)
                        for item in result
                    ]
                return []
            except Exception as e:
                wait = self.RETRY_BACKOFF ** attempt
                logger.warning(f"Firecrawl map attempt {attempt + 1} failed for {url}: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(wait)
        return []
