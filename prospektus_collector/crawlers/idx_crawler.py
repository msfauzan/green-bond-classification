"""IDX (Indonesia Stock Exchange) crawler for prospektus announcements."""
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base_crawler import BaseCrawler
from ..tracker import ProspektusTracker
from ..config import POJK_18_DATE


class IDXCrawler(BaseCrawler):
    """Crawl IDX website for prospektus announcements."""

    # IDX URLs
    IDX_ANNOUNCEMENTS_URL = "https://www.idx.co.id/id/berita/pengumuman"
    IDX_SEARCH_URL = "https://www.idx.co.id/id/berita/pengumuman?search=prospektus"
    IDX_LISTED_COMPANIES = "https://www.idx.co.id/id/perusahaan-tercatat/daftar-perusahaan-tercatat"

    def __init__(self, tracker: Optional[ProspektusTracker] = None):
        super().__init__(source_name="idx")
        self.tracker = tracker or ProspektusTracker()

    def run(self, max_pages: int = 5) -> List[Dict[str, Any]]:
        """Crawl IDX for prospektus announcements."""
        results = []

        print(f"\n{'='*50}")
        print(f"[IDX] Starting crawl...")
        print(f"{'='*50}")

        # Scrape main announcements page
        scraped = self.scrape_url(self.IDX_ANNOUNCEMENTS_URL)

        if not scraped:
            print("[IDX] Failed to scrape IDX website")
            return results

        # Get content from markdown and HTML
        content = scraped.get('markdown', '') + scraped.get('html', '')

        # Extract PDF links
        pdf_links = self.extract_pdf_links(content, self.IDX_ANNOUNCEMENTS_URL)

        print(f"[IDX] Found {len(pdf_links)} PDF links")

        # Filter and process
        for pdf_info in pdf_links:
            url = pdf_info['url']

            # Skip if already processed
            if self.tracker.exists(url):
                print(f"[IDX] Skipping (already processed): {pdf_info['filename']}")
                continue

            # Extract emiten code from filename or URL
            emiten_code = self._extract_emiten_code(pdf_info['filename'], url)

            results.append({
                "url": url,
                "filename": pdf_info['filename'],
                "emiten_code": emiten_code,
                "source": "idx"
            })

        print(f"[IDX] Found {len(results)} new PDFs to download")
        return results

    def _extract_emiten_code(self, filename: str, url: str) -> str:
        """Extract emiten code from filename or URL."""
        # Common patterns in IDX filenames
        patterns = [
            r'([A-Z]{4})[_\-\s]',  # CODE_filename
            r'_([A-Z]{4})_',        # _CODE_
            r'/([A-Z]{4})/',        # /CODE/
            r'\(([A-Z]{4})\)',      # (CODE)
        ]

        for pattern in patterns:
            match = re.search(pattern, filename)
            if match:
                return match.group(1)

        # Try URL
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return "UNKNOWN"

    def search_by_emiten(self, emiten_code: str) -> List[Dict[str, Any]]:
        """Search for prospektus by emiten code."""
        search_url = f"https://www.idx.co.id/id/berita/pengumuman?search={emiten_code}"

        scraped = self.scrape_url(search_url)
        if not scraped:
            return []

        content = scraped.get('markdown', '') + scraped.get('html', '')
        pdf_links = self.extract_pdf_links(content, search_url)

        results = []
        for pdf_info in pdf_links:
            if not self.tracker.exists(pdf_info['url']):
                results.append({
                    "url": pdf_info['url'],
                    "filename": pdf_info['filename'],
                    "emiten_code": emiten_code,
                    "source": "idx"
                })

        return results

    def search_by_keywords(self, keywords: List[str]) -> List[Dict[str, Any]]:
        """Search for prospektus by keywords."""
        all_results = []

        for keyword in keywords:
            search_url = f"https://www.idx.co.id/id/berita/pengumuman?search={keyword}"

            scraped = self.scrape_url(search_url)
            if not scraped:
                continue

            content = scraped.get('markdown', '') + scraped.get('html', '')
            pdf_links = self.extract_pdf_links(content, search_url)

            for pdf_info in pdf_links:
                if not self.tracker.exists(pdf_info['url']):
                    emiten_code = self._extract_emiten_code(pdf_info['filename'], pdf_info['url'])
                    all_results.append({
                        "url": pdf_info['url'],
                        "filename": pdf_info['filename'],
                        "emiten_code": emiten_code,
                        "source": "idx"
                    })

        return all_results

    def download_all(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Download all PDFs from results."""
        import asyncio

        downloaded = []

        for item in results:
            filepath = asyncio.run(
                self.download_pdf(
                    item['url'],
                    item['filename'],
                    item['emiten_code']
                )
            )

            if filepath:
                item['filepath'] = filepath
                item['status'] = 'downloaded'
                downloaded.append(item)

                # Track in database
                self.tracker.add(
                    url=item['url'],
                    filename=item['filename'],
                    source=item['source'],
                    emiten_code=item['emiten_code'],
                    status='completed'
                )
            else:
                item['status'] = 'failed'
                self.tracker.add(
                    url=item['url'],
                    filename=item['filename'],
                    source=item['source'],
                    emiten_code=item['emiten_code'],
                    status='failed',
                    error_message='Download failed'
                )

        return downloaded
