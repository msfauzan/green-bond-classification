"""Base crawler using Firecrawl API."""
import asyncio
import aiohttp
import aiofiles
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

try:
    from firecrawl import FirecrawlApp
except ImportError:
    FirecrawlApp = None

from ..config import (
    FIRECRAWL_API_KEY,
    FIRECRAWL_TIMEOUT,
    PDF_KEYWORDS,
    POJK_18_DATE,
    BASE_DIR
)


class BaseCrawler:
    """Base class for PDF crawlers using Firecrawl."""

    def __init__(self, source_name: str, output_dir: Optional[Path] = None):
        self.source_name = source_name
        self.output_dir = output_dir or BASE_DIR / "Prospektus_Downloaded" / source_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if FirecrawlApp and FIRECRAWL_API_KEY:
            self.firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY)
        else:
            self.firecrawl = None
            print("Warning: Firecrawl not initialized. Check FIRECRAWL_API_KEY.")

    def scrape_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape a single URL using Firecrawl."""
        if not self.firecrawl:
            raise RuntimeError("Firecrawl not initialized")

        try:
            result = self.firecrawl.scrape_url(url, params={
                'formats': ['markdown', 'html'],
            })
            return result
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None

    def crawl_url(self, url: str, limit: int = 10) -> Optional[List[Dict]]:
        """Crawl a URL and its links using Firecrawl."""
        if not self.firecrawl:
            raise RuntimeError("Firecrawl not initialized")

        try:
            result = self.firecrawl.crawl_url(url, params={
                'limit': limit,
                'scrapeOptions': {
                    'formats': ['markdown']
                }
            })
            return result.get('data', [])
        except Exception as e:
            print(f"Error crawling {url}: {e}")
            return None

    def extract_pdf_links(self, content: str, base_url: str = "") -> List[Dict[str, str]]:
        """Extract PDF links from scraped content."""
        pdf_links = []

        # Pattern for PDF URLs
        pdf_pattern = r'https?://[^\s<>"]+?\.pdf'
        matches = re.findall(pdf_pattern, content, re.IGNORECASE)

        for url in matches:
            # Clean URL (remove trailing punctuation)
            clean_url = url.rstrip('.,;:)')

            # Extract filename
            filename = os.path.basename(clean_url.split('?')[0])

            # Check if filename contains relevant keywords
            if self._is_relevant_pdf(filename):
                pdf_links.append({
                    "url": clean_url,
                    "filename": filename
                })

        return pdf_links

    def _is_relevant_pdf(self, filename: str) -> bool:
        """Check if PDF filename suggests it's a prospektus."""
        filename_lower = filename.lower()

        # Must contain at least one keyword
        for keyword in PDF_KEYWORDS:
            if keyword.lower() in filename_lower:
                return True

        # Check for common prospektus patterns
        patterns = [
            r'prospektus',
            r'pmhmetd',
            r'pub[_-]?\d+',
            r'obligasi',
            r'sukuk',
            r'\d{8}[_-][A-Z]{4}'  # Date code pattern
        ]

        for pattern in patterns:
            if re.search(pattern, filename_lower):
                return True

        return False

    async def download_pdf(
        self,
        url: str,
        filename: str,
        emiten_code: str = "UNKNOWN"
    ) -> Optional[str]:
        """Download PDF asynchronously."""
        # Generate standardized filename
        date_prefix = datetime.now().strftime("%Y%m%d")
        safe_filename = f"{date_prefix}_{emiten_code}_{filename}"
        safe_filename = re.sub(r'[^\w\-.]', '_', safe_filename)

        filepath = self.output_dir / safe_filename

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        async with aiofiles.open(filepath, 'wb') as f:
                            await f.write(await response.read())
                        print(f"Downloaded: {safe_filename}")
                        return str(filepath)
                    else:
                        print(f"Failed to download {url}: HTTP {response.status}")
                        return None
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return None

    def run(self) -> List[Dict[str, Any]]:
        """Run the crawler. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement run()")
