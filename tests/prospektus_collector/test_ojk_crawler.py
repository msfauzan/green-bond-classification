import pytest
import asyncio
from unittest.mock import MagicMock
from prospektus_collector.crawlers.ojk_crawler import OJKCrawler, is_debt_security_prospektus


def test_is_debt_security_prospektus_true():
    assert is_debt_security_prospektus("prospektus_obligasi_ABCD.pdf") is True
    assert is_debt_security_prospektus("Penyampaian Prospektus Sukuk XYZ.pdf") is True
    assert is_debt_security_prospektus("penawaran_umum_mtn.pdf") is True
    assert is_debt_security_prospektus("prospektus eba xyz.pdf") is True


def test_is_debt_security_prospektus_false():
    assert is_debt_security_prospektus("laporan_tahunan_2023.pdf") is False
    assert is_debt_security_prospektus("foto_kegiatan.jpg") is False
    assert is_debt_security_prospektus("rups_2024.pdf") is False


def test_ojk_crawler_run_filters_prospektus_only():
    crawler = OJKCrawler.__new__(OJKCrawler)
    crawler.map_site = MagicMock(return_value=[
        "https://ojk.go.id/files/prospektus_obligasi_ABCD_2024.pdf",
        "https://ojk.go.id/files/laporan_keuangan.pdf",
        "https://ojk.go.id/files/prospektus_sukuk_EFGH_2024.pdf",
    ])

    async def run():
        return await crawler.run(MagicMock())

    result = asyncio.get_event_loop().run_until_complete(run())
    urls = [r["url"] for r in result]
    assert "https://ojk.go.id/files/prospektus_obligasi_ABCD_2024.pdf" in urls
    assert "https://ojk.go.id/files/prospektus_sukuk_EFGH_2024.pdf" in urls
    assert "https://ojk.go.id/files/laporan_keuangan.pdf" not in urls


def test_ojk_crawler_run_result_structure():
    crawler = OJKCrawler.__new__(OJKCrawler)
    crawler.map_site = MagicMock(return_value=[
        "https://ojk.go.id/files/prospektus_obligasi_ABCD_2024.pdf",
    ])

    async def run():
        return await crawler.run(MagicMock())

    result = asyncio.get_event_loop().run_until_complete(run())
    assert len(result) == 1
    item = result[0]
    assert "url" in item
    assert "filename" in item
    assert "emiten_code" in item
    assert "source" in item
    assert item["source"] == "ojk"
