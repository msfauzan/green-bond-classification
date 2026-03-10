import pytest
from unittest.mock import MagicMock
from prospektus_collector.crawlers.idx_crawler import IDXCrawler, filter_pdf_links_by_keywords


def test_filter_pdf_links_by_keywords_matches():
    links = [
        "https://idx.co.id/prospektus_ABCD.pdf",
        "https://idx.co.id/laporan_keuangan.pdf",
        "https://idx.co.id/penawaran_umum_EFGH.pdf",
        "https://idx.co.id/foto_rups.jpg",
    ]
    result = filter_pdf_links_by_keywords(links, ["prospektus", "penawaran_umum"])
    assert "https://idx.co.id/prospektus_ABCD.pdf" in result
    assert "https://idx.co.id/penawaran_umum_EFGH.pdf" in result
    assert "https://idx.co.id/laporan_keuangan.pdf" not in result
    assert "https://idx.co.id/foto_rups.jpg" not in result


def test_filter_pdf_links_case_insensitive():
    links = ["https://idx.co.id/PROSPEKTUS_ABCD.PDF"]
    result = filter_pdf_links_by_keywords(links, ["prospektus"])
    assert len(result) == 1


def test_idx_crawler_discover_emiten_extracts_kode():
    """discover_emiten harus mengekstrak kode emiten dari URL profil IDX."""
    crawler = IDXCrawler.__new__(IDXCrawler)
    crawler.fc = MagicMock()
    crawler.MAX_RETRIES = 1
    crawler.RETRY_BACKOFF = 0
    crawler.timeout = 30

    crawler.map_site = MagicMock(return_value=[
        "https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat/?kodeEmiten=ABCD",
        "https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat/?kodeEmiten=EFGH",
        "https://www.idx.co.id/id/page-lain",
    ])
    crawler.scrape_page = MagicMock(return_value={
        "links": ["https://www.abcd.co.id"],
        "markdown": "Website: https://www.abcd.co.id",
    })

    emiten = crawler.discover_emiten()
    kodes = [e["kode"] for e in emiten]
    assert "ABCD" in kodes
    assert "EFGH" in kodes


def test_idx_crawler_discover_emiten_structure():
    """Setiap item dari discover_emiten harus punya key 'kode' dan 'website'."""
    crawler = IDXCrawler.__new__(IDXCrawler)
    crawler.fc = MagicMock()
    crawler.MAX_RETRIES = 1
    crawler.RETRY_BACKOFF = 0
    crawler.timeout = 30

    crawler.map_site = MagicMock(return_value=[
        "https://www.idx.co.id/?kodeEmiten=ABCD",
    ])
    crawler.scrape_page = MagicMock(return_value={
        "links": ["https://www.abcd.co.id"],
        "markdown": "",
    })

    emiten = crawler.discover_emiten()
    assert len(emiten) == 1
    assert "kode" in emiten[0]
    assert "website" in emiten[0]
    assert emiten[0]["kode"] == "ABCD"


@pytest.mark.asyncio
async def test_idx_crawler_run_returns_list():
    import aiohttp
    crawler = IDXCrawler.__new__(IDXCrawler)
    crawler.fc = MagicMock()
    crawler.MAX_RETRIES = 1
    crawler.RETRY_BACKOFF = 0
    crawler.timeout = 30

    crawler.map_site = MagicMock(return_value=[
        "https://idx.co.id/files/prospektus_ABCD_2024.pdf",
        "https://idx.co.id/files/laporan_keuangan.pdf",
    ])
    mock_session = MagicMock()

    result = await crawler.run(mock_session)
    assert isinstance(result, list)
    # Hanya PDF prospektus yang lolos filter
    urls = [r["url"] for r in result]
    assert "https://idx.co.id/files/prospektus_ABCD_2024.pdf" in urls
    assert "https://idx.co.id/files/laporan_keuangan.pdf" not in urls
