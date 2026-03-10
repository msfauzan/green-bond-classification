import pytest
import asyncio
from unittest.mock import MagicMock
from prospektus_collector.crawlers.emiten_crawler import EmitenCrawler, find_ir_url


def test_find_ir_url_matches_investor_relations():
    urls = [
        "https://company.co.id/about",
        "https://company.co.id/investor-relations/publications",
        "https://company.co.id/news",
    ]
    result = find_ir_url(urls)
    assert result == "https://company.co.id/investor-relations/publications"


def test_find_ir_url_returns_none_if_not_found():
    urls = ["https://company.co.id/about", "https://company.co.id/contact"]
    result = find_ir_url(urls)
    assert result is None


def test_find_ir_url_hubungan_investor():
    urls = ["https://co.id/hubungan-investor/prospektus", "https://co.id/about"]
    assert find_ir_url(urls) == "https://co.id/hubungan-investor/prospektus"


def test_find_ir_url_publikasi():
    urls = ["https://co.id/publikasi", "https://co.id/home"]
    assert find_ir_url(urls) is not None


def test_emiten_crawler_run_skips_emiten_without_website():
    crawler = EmitenCrawler.__new__(EmitenCrawler)
    crawler.map_site = MagicMock()
    crawler.scrape_page = MagicMock()

    async def run():
        emiten_list = [{"kode": "ABCD", "website": None}]
        return await crawler.run(emiten_list, MagicMock())

    result = asyncio.get_event_loop().run_until_complete(run())
    assert result == []
    crawler.map_site.assert_not_called()


def test_emiten_crawler_run_finds_pdf():
    crawler = EmitenCrawler.__new__(EmitenCrawler)
    crawler.MAX_RETRIES = 1
    crawler.RETRY_BACKOFF = 0
    crawler.timeout = 30
    crawler.fc = MagicMock()

    crawler.map_site = MagicMock(return_value=[
        "https://emiten.co.id/investor-relations",
        "https://emiten.co.id/investor-relations/prospektus_2024.pdf",
    ])
    crawler.scrape_page = MagicMock(return_value={
        "links": ["https://emiten.co.id/investor-relations/prospektus_obligasi_2024.pdf"],
        "markdown": "Prospektus Obligasi 2024",
    })

    async def run():
        emiten_list = [{"kode": "ABCD", "website": "https://emiten.co.id"}]
        return await crawler.run(emiten_list, MagicMock())

    result = asyncio.get_event_loop().run_until_complete(run())
    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0]["emiten_code"] == "ABCD"
    assert result[0]["source"] == "emiten"
