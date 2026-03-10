import pytest
from unittest.mock import MagicMock, patch
from prospektus_collector.crawlers.idx_crawler import (
    IDXCrawler,
    filter_pdf_links_by_keywords,
    extract_prospektus_links_from_markdown,
)


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


def test_extract_prospektus_links_from_markdown():
    """extract_prospektus_links_from_markdown mendeteksi PDF dari link text."""
    markdown = (
        "- [Prospektus Obligasi ABCD 2024](https://idx.co.id/a1b2c3.pdf)\n"
        "- [Laporan Keuangan EFGH](https://idx.co.id/lk_efgh.pdf)\n"
        "- [Penawaran Umum Sukuk IJKL](https://idx.co.id/d4e5f6.pdf)\n"
        "- [Info Umum](https://idx.co.id/page)\n"
    )
    result = extract_prospektus_links_from_markdown(markdown, ["prospektus", "sukuk"])
    urls = [url for _, url in result]
    assert "https://idx.co.id/a1b2c3.pdf" in urls
    assert "https://idx.co.id/d4e5f6.pdf" in urls
    assert "https://idx.co.id/lk_efgh.pdf" not in urls
    assert "https://idx.co.id/page" not in urls


def test_idx_crawler_discover_emiten_extracts_kode():
    """discover_emiten harus mengekstrak kode emiten dari URL profil IDX (path-based)."""
    crawler = IDXCrawler.__new__(IDXCrawler)
    crawler.fc = MagicMock()
    crawler.MAX_RETRIES = 1
    crawler.RETRY_BACKOFF = 0
    crawler.timeout = 30

    # New implementation uses _map_emiten_with_limit, not map_site
    crawler._map_emiten_with_limit = MagicMock(return_value=[
        "https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat/ABCD",
        "https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat/EFGH",
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

    crawler._map_emiten_with_limit = MagicMock(return_value=[
        "https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan-tercatat/ABCD",
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
    """run() harus mengembalikan PDF list dari markdown parsing announcement page."""
    crawler = IDXCrawler.__new__(IDXCrawler)
    crawler.fc = MagicMock()
    crawler.MAX_RETRIES = 1
    crawler.RETRY_BACKOFF = 0
    crawler.timeout = 30

    # New implementation uses scrape_page and parses markdown link text
    crawler.scrape_page = MagicMock(return_value={
        "links": [],
        "markdown": (
            "- [Prospektus Obligasi ABCD 2024-05-01](https://idx.co.id/hashed_abc.pdf)\n"
            "- [Laporan Keuangan XYZ](https://idx.co.id/lk_xyz.pdf)\n"
        ),
    })
    mock_session = MagicMock()

    result = await crawler.run(mock_session)
    assert isinstance(result, list)
    # Hanya PDF yang link text-nya mengandung keyword yang lolos
    urls = [r["url"] for r in result]
    assert "https://idx.co.id/hashed_abc.pdf" in urls
    assert "https://idx.co.id/lk_xyz.pdf" not in urls
