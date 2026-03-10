"""
Orchestrator untuk Prospektus Collector.
Entry point: python -m prospektus_collector.orchestrator

Usage:
    python -m prospektus_collector.orchestrator
    python -m prospektus_collector.orchestrator --source idx
    python -m prospektus_collector.orchestrator --source ojk,ksei
    python -m prospektus_collector.orchestrator --retry-failed
"""
import asyncio
import logging
import argparse
from typing import Literal

import aiohttp

from .config import (
    FIRECRAWL_API_KEY,
    POJK_18_DATE,
    R2_BUCKET_NAME,
    R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY,
    R2_ENDPOINT_URL,
)
from .tracker import Tracker
from .uploaders.r2_uploader import R2Uploader
from .crawlers.base_crawler import BaseCrawler
from .crawlers.idx_crawler import IDXCrawler
from .crawlers.ojk_crawler import OJKCrawler
from .crawlers.ksei_crawler import KSEICrawler
from .crawlers.emiten_crawler import EmitenCrawler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("collector.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def build_pdf_pipeline(
    item: dict,
    tracker: Tracker,
    uploader: R2Uploader,
) -> Literal["skip", "process"]:
    """Cek apakah PDF perlu diproses atau sudah ada (duplikat)."""
    if tracker.is_seen(item["url"]):
        return "skip"
    return "process"


def print_summary(stats: dict) -> None:
    total_found = sum(s["found"] for s in stats.values())
    total_uploaded = sum(s["uploaded"] for s in stats.values())
    total_skipped = sum(s["skipped"] for s in stats.values())
    total_failed = sum(s["failed"] for s in stats.values())

    print("\n" + "=" * 52)
    print("=== PROSPEKTUS COLLECTOR ===")
    print(f"Filter tanggal : >= {POJK_18_DATE} (POJK 18/2023)\n")
    labels = {"idx": "IDX", "ojk": "OJK", "ksei": "KSEI", "emiten": "Emiten"}
    for source, label in labels.items():
        if source in stats:
            s = stats[source]
            print(
                f"[{label:<7}] Ditemukan {s['found']:>3} PDF | "
                f"Diupload {s['uploaded']:>3} | "
                f"Diskip {s['skipped']:>3} | "
                f"Gagal {s['failed']:>3}"
            )
    print()
    print("=== RINGKASAN TOTAL ===")
    print(f"Total Ditemukan : {total_found}")
    print(f"Diupload        : {total_uploaded}")
    print(f"Diskip          : {total_skipped} (duplikat)")
    print(f"Gagal           : {total_failed} (lihat collector.log)")
    print("=" * 52)


async def process_pdf_items(
    items: list,
    session: aiohttp.ClientSession,
    tracker: Tracker,
    uploader: R2Uploader,
    stats: dict,
    downloader: BaseCrawler,
) -> None:
    """Download dan upload satu batch PDF items."""
    for item in items:
        source = item["source"]
        # Guard: pastikan source ada di stats
        if source not in stats:
            stats[source] = {"found": 0, "uploaded": 0, "skipped": 0, "failed": 0}

        decision = build_pdf_pipeline(item, tracker, uploader)

        if decision == "skip":
            stats[source]["skipped"] += 1
            logger.info(f"SKIP (duplicate): {item['url']}")
            continue

        stats[source]["found"] += 1
        pdf_bytes = await downloader.download_pdf(item["url"], session)

        if not pdf_bytes:
            stats[source]["failed"] += 1
            tracker.mark(
                url=item["url"],
                filename=item["filename"],
                source=source,
                emiten_code=item["emiten_code"],
                r2_key="",
                status="failed",
            )
            continue

        r2_key = uploader.build_r2_key(source, item["emiten_code"], item["filename"])
        try:
            uploader.upload(pdf_bytes, r2_key)
            tracker.mark(
                url=item["url"],
                filename=item["filename"],
                source=source,
                emiten_code=item["emiten_code"],
                r2_key=r2_key,
                status="uploaded",
            )
            stats[source]["uploaded"] += 1
            logger.info(f"UPLOADED: {r2_key}")
        except Exception as e:
            logger.error(f"R2 upload failed for {r2_key}: {e}")
            stats[source]["failed"] += 1
            tracker.mark(
                url=item["url"],
                filename=item["filename"],
                source=source,
                emiten_code=item["emiten_code"],
                r2_key="",
                status="failed",
            )


async def main(sources: list, retry_failed: bool = False) -> None:
    tracker = Tracker()
    uploader = R2Uploader(
        bucket=R2_BUCKET_NAME,
        endpoint_url=R2_ENDPOINT_URL,
        access_key=R2_ACCESS_KEY_ID,
        secret_key=R2_SECRET_ACCESS_KEY,
    )
    # BaseCrawler hanya untuk download PDF
    downloader = BaseCrawler(firecrawl_api_key=FIRECRAWL_API_KEY)

    stats: dict = {
        s: {"found": 0, "uploaded": 0, "skipped": 0, "failed": 0}
        for s in sources
    }

    idx_crawler = IDXCrawler(firecrawl_api_key=FIRECRAWL_API_KEY)
    ojk_crawler = OJKCrawler(firecrawl_api_key=FIRECRAWL_API_KEY)
    ksei_crawler = KSEICrawler(firecrawl_api_key=FIRECRAWL_API_KEY)
    emiten_crawler = EmitenCrawler(firecrawl_api_key=FIRECRAWL_API_KEY)

    async with aiohttp.ClientSession() as session:
        # Step 1: Discover emiten dari IDX
        emiten_list = []
        if "idx" in sources or "emiten" in sources:
            logger.info("Discovering emiten list from IDX...")
            emiten_list = idx_crawler.discover_emiten()

        # Step 2: Jalankan semua crawler secara concurrent
        crawler_tasks = []
        source_order = []
        if "idx" in sources:
            crawler_tasks.append(idx_crawler.run(session))
            source_order.append("idx")
        if "ojk" in sources:
            crawler_tasks.append(ojk_crawler.run(session))
            source_order.append("ojk")
        if "ksei" in sources:
            crawler_tasks.append(ksei_crawler.run(session))
            source_order.append("ksei")
        if "emiten" in sources and emiten_list:
            crawler_tasks.append(emiten_crawler.run(emiten_list, session))
            source_order.append("emiten")

        crawl_results = await asyncio.gather(*crawler_tasks, return_exceptions=True)

        # Step 3: Proses hasil setiap crawler
        for source, result in zip(source_order, crawl_results):
            if isinstance(result, Exception):
                logger.error(f"Crawler {source} failed: {result}")
                continue
            await process_pdf_items(result, session, tracker, uploader, stats, downloader)

        # Step 4: Retry failed jika diminta
        if retry_failed:
            failed_items = tracker.get_failed()
            logger.info(f"Retrying {len(failed_items)} failed downloads...")
            retry_list = [
                {
                    "url": item["url"],
                    "filename": item["filename"],
                    "source": item["source"],
                    "emiten_code": item["emiten_code"],
                    "date": "",
                }
                for item in failed_items
            ]
            for item in failed_items:
                tracker.update_status(item["url"], "retrying")
            await process_pdf_items(retry_list, session, tracker, uploader, stats, downloader)

    print_summary(stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Prospektus Collector — POJK 18/2023")
    parser.add_argument(
        "--source",
        default="idx,ojk,ksei,emiten",
        help="Sumber yang di-crawl (comma-separated): idx,ojk,ksei,emiten",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry PDF yang gagal didownload sebelumnya",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sources = [s.strip() for s in args.source.split(",")]
    asyncio.run(main(sources=sources, retry_failed=args.retry_failed))
