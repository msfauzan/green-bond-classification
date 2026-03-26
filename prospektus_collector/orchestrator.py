"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           PROSPEKTUS ORCHESTRATOR                                             ║
║           Coordinated collection from multiple sources                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
import asyncio
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from .tracker import ProspektusTracker
from .crawlers.idx_crawler import IDXCrawler
from .uploaders.r2_uploader import R2Uploader
from .config import BASE_DIR, POJK_18_DATE


class ProspektusOrchestrator:
    """Orchestrate prospektus collection from multiple sources."""

    def __init__(self, upload_to_r2: bool = True, output_dir: Optional[Path] = None):
        self.tracker = ProspektusTracker()
        self.upload_to_r2 = upload_to_r2
        self.output_dir = output_dir or BASE_DIR / "Prospektus_Downloaded"

        # Initialize R2 uploader
        self.r2_uploader = None
        if upload_to_r2:
            try:
                self.r2_uploader = R2Uploader()
                print("✅ R2 uploader initialized")
            except ValueError as e:
                print(f"⚠️ R2 uploader not available: {e}")
                self.upload_to_r2 = False

    def run(
        self,
        sources: Optional[List[str]] = None,
        retry_failed: bool = False,
        emiten_codes: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run the collection pipeline."""
        print("\n" + "=" * 60)
        print("🚀 PROSPEKTUS COLLECTOR")
        print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Filter: Documents >= {POJK_18_DATE}")
        print("=" * 60)

        sources = sources or ['idx']
        stats = {
            'started_at': datetime.now().isoformat(),
            'total_found': 0,
            'total_downloaded': 0,
            'total_uploaded': 0,
            'total_skipped': 0,
            'total_failed': 0,
            'by_source': {}
        }

        # Retry failed if requested
        if retry_failed:
            return self._retry_failed()

        # Run each source crawler
        for source in sources:
            print(f"\n{'='*50}")
            print(f"[{source.upper()}] Starting crawl...")
            print(f"{'='*50}")

            source_stats = {
                'found': 0,
                'downloaded': 0,
                'uploaded': 0,
                'skipped': 0,
                'failed': 0
            }

            try:
                if source == 'idx':
                    results = self._crawl_idx(emiten_codes, keywords)
                else:
                    print(f"⚠️ Unknown source: {source}")
                    continue

                source_stats['found'] = len(results)
                stats['total_found'] += len(results)

                # Process results
                for item in results:
                    result = self._process_item(item, source)

                    if result == 'downloaded':
                        source_stats['downloaded'] += 1
                        stats['total_downloaded'] += 1
                    elif result == 'uploaded':
                        source_stats['downloaded'] += 1
                        source_stats['uploaded'] += 1
                        stats['total_downloaded'] += 1
                        stats['total_uploaded'] += 1
                    elif result == 'skipped':
                        source_stats['skipped'] += 1
                        stats['total_skipped'] += 1
                    else:
                        source_stats['failed'] += 1
                        stats['total_failed'] += 1

            except Exception as e:
                print(f"❌ Error crawling {source}: {e}")
                source_stats['error'] = str(e)

            stats['by_source'][source] = source_stats

        stats['completed_at'] = datetime.now().isoformat()

        # Print summary
        self._print_summary(stats)

        return stats

    def _crawl_idx(
        self,
        emiten_codes: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Crawl IDX source."""
        crawler = IDXCrawler(tracker=self.tracker)

        if emiten_codes:
            # Search by specific emiten codes
            results = []
            for code in emiten_codes:
                results.extend(crawler.search_by_emiten(code))
            return results
        elif keywords:
            # Search by keywords
            return crawler.search_by_keywords(keywords)
        else:
            # General crawl
            return crawler.run()

    def _process_item(self, item: Dict[str, Any], source: str) -> str:
        """Process a single item. Returns status."""
        url = item['url']
        filename = item['filename']
        emiten_code = item.get('emiten_code', 'UNKNOWN')

        # Check if already processed
        if self.tracker.exists(url):
            print(f"⏭️ Skipping (exists): {filename}")
            return 'skipped'

        # Check if already in R2
        if self.upload_to_r2 and self.r2_uploader:
            if self.r2_uploader.file_exists(source, emiten_code, filename):
                print(f"⏭️ Skipping (in R2): {filename}")
                self.tracker.add(
                    url=url,
                    filename=filename,
                    source=source,
                    emiten_code=emiten_code,
                    r2_key=f"prospektus/{source}/{emiten_code}/{filename}",
                    status='completed'
                )
                return 'skipped'

        # Download
        crawler = IDXCrawler(tracker=self.tracker)
        filepath = asyncio.run(crawler.download_pdf(url, filename, emiten_code))

        if not filepath:
            self.tracker.add(
                url=url,
                filename=filename,
                source=source,
                emiten_code=emiten_code,
                status='failed',
                error_message='Download failed'
            )
            return 'failed'

        # Upload to R2
        r2_key = None
        if self.upload_to_r2 and self.r2_uploader:
            r2_key = self.r2_uploader.upload_file(filepath, source, emiten_code)

        # Track
        self.tracker.add(
            url=url,
            filename=filename,
            source=source,
            emiten_code=emiten_code,
            r2_key=r2_key,
            status='completed'
        )

        return 'uploaded' if r2_key else 'downloaded'

    def _retry_failed(self) -> Dict[str, Any]:
        """Retry failed downloads."""
        failed = self.tracker.get_failed()
        print(f"\n🔄 Retrying {len(failed)} failed downloads...")

        stats = {
            'started_at': datetime.now().isoformat(),
            'total_found': len(failed),
            'total_downloaded': 0,
            'total_uploaded': 0,
            'total_failed': 0,
            'by_source': {}
        }

        for item in failed:
            result = self._process_item(item, item['source'])
            if result in ['downloaded', 'uploaded']:
                stats['total_downloaded'] += 1
                if result == 'uploaded':
                    stats['total_uploaded'] += 1
            else:
                stats['total_failed'] += 1

        stats['completed_at'] = datetime.now().isoformat()
        self._print_summary(stats)
        return stats

    def _print_summary(self, stats: Dict[str, Any]):
        """Print collection summary."""
        print("\n" + "=" * 60)
        print("📊 COLLECTION SUMMARY")
        print("=" * 60)

        for source, source_stats in stats.get('by_source', {}).items():
            print(f"\n[{source.upper()}]")
            print(f"  Found:      {source_stats.get('found', 0)}")
            print(f"  Downloaded: {source_stats.get('downloaded', 0)}")
            print(f"  Uploaded:   {source_stats.get('uploaded', 0)}")
            print(f"  Skipped:    {source_stats.get('skipped', 0)}")
            print(f"  Failed:     {source_stats.get('failed', 0)}")

        print("\n[TOTAL]")
        print(f"  Found:      {stats.get('total_found', 0)}")
        print(f"  Downloaded: {stats.get('total_downloaded', 0)}")
        print(f"  Uploaded:   {stats.get('total_uploaded', 0)}")
        print(f"  Skipped:    {stats.get('total_skipped', 0)}")
        print(f"  Failed:     {stats.get('total_failed', 0)}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Collect prospektus from various sources',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m prospektus_collector.orchestrator --source idx
  python -m prospektus_collector.orchestrator --source idx --emiten BBCA BBRI
  python -m prospektus_collector.orchestrator --source idx --keywords "green bond"
  python -m prospektus_collector.orchestrator --retry-failed
  python -m prospektus_collector.orchestrator --source idx --no-upload
        """
    )
    parser.add_argument(
        '--source', type=str, default='idx',
        help='Source to crawl (idx, all)'
    )
    parser.add_argument(
        '--emiten', nargs='+', type=str,
        help='Specific emiten codes to search'
    )
    parser.add_argument(
        '--keywords', nargs='+', type=str,
        help='Keywords to search for'
    )
    parser.add_argument(
        '--retry-failed', action='store_true',
        help='Retry failed downloads'
    )
    parser.add_argument(
        '--no-upload', action='store_true',
        help='Skip R2 upload'
    )

    args = parser.parse_args()

    sources = ['idx'] if args.source == 'idx' else ['idx']

    orchestrator = ProspektusOrchestrator(upload_to_r2=not args.no_upload)
    orchestrator.run(
        sources=sources,
        retry_failed=args.retry_failed,
        emiten_codes=args.emiten,
        keywords=args.keywords
    )


if __name__ == "__main__":
    main()
