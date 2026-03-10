import pytest
from unittest.mock import AsyncMock, MagicMock
from prospektus_collector.crawlers.base_crawler import BaseCrawler, is_pdf_link, extract_date_from_text


def test_is_pdf_link_true():
    assert is_pdf_link("https://example.com/prospektus.pdf") is True
    assert is_pdf_link("https://example.com/file.PDF") is True


def test_is_pdf_link_false():
    assert is_pdf_link("https://example.com/page.html") is False
    assert is_pdf_link("https://example.com/image.png") is False


def test_is_pdf_link_with_query_params():
    assert is_pdf_link("https://example.com/download?file=prospektus.pdf") is True


def test_extract_date_from_text_iso():
    assert extract_date_from_text("Dokumen 2024-03-15 resmi") == "2024-03-15"


def test_extract_date_from_text_slash():
    assert extract_date_from_text("15/03/2024 diumumkan") == "2024-03-15"


def test_extract_date_from_text_none():
    assert extract_date_from_text("Tidak ada tanggal disini") is None


@pytest.mark.asyncio
async def test_base_crawler_download_pdf_success():
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.read = AsyncMock(return_value=b"%PDF-1.4 content")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    crawler = BaseCrawler(firecrawl_api_key="test-key")
    result = await crawler.download_pdf("https://example.com/doc.pdf", session=mock_session)
    assert result == b"%PDF-1.4 content"


@pytest.mark.asyncio
async def test_base_crawler_download_pdf_returns_none_on_404():
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.get.return_value = mock_response

    crawler = BaseCrawler(firecrawl_api_key="test-key")
    result = await crawler.download_pdf("https://example.com/notfound.pdf", session=mock_session, retries=1)
    assert result is None
