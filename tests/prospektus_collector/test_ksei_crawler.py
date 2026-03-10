import pytest
import asyncio
from unittest.mock import MagicMock
from prospektus_collector.crawlers.ksei_crawler import KSEICrawler, is_ksei_prospektus


def test_is_ksei_prospektus_true():
    assert is_ksei_prospektus("prospektus_ABCD.pdf") is True
    assert is_ksei_prospektus("penawaran_umum_obligasi.pdf") is True
    assert is_ksei_prospektus("sukuk_xyz_2024.pdf") is True


def test_is_ksei_prospektus_false():
    assert is_ksei_prospektus("laporan_keuangan.pdf") is False
    assert is_ksei_prospektus("rups_notulensi.pdf") is False


def test_ksei_crawler_run_returns_list():
    crawler = KSEICrawler.__new__(KSEICrawler)
    crawler.map_site = MagicMock(return_value=[
        "https://ksei.co.id/files/prospektus_ABCD_2024.pdf",
        "https://ksei.co.id/files/laporan.pdf",
    ])

    async def run():
        return await crawler.run(MagicMock())

    result = asyncio.get_event_loop().run_until_complete(run())
    assert isinstance(result, list)
    urls = [r["url"] for r in result]
    assert "https://ksei.co.id/files/prospektus_ABCD_2024.pdf" in urls
    assert "https://ksei.co.id/files/laporan.pdf" not in urls


def test_ksei_crawler_result_structure():
    crawler = KSEICrawler.__new__(KSEICrawler)
    crawler.map_site = MagicMock(return_value=[
        "https://ksei.co.id/files/prospektus_ABCD_2024.pdf",
    ])

    async def run():
        return await crawler.run(MagicMock())

    result = asyncio.get_event_loop().run_until_complete(run())
    assert result[0]["source"] == "ksei"
