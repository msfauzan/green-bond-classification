import pytest
from unittest.mock import MagicMock
from prospektus_collector.orchestrator import build_pdf_pipeline, print_summary


def test_print_summary_outputs_all_sources(capsys):
    stats = {
        "idx":    {"found": 10, "uploaded": 9, "skipped": 1, "failed": 0},
        "ojk":    {"found":  5, "uploaded": 5, "skipped": 0, "failed": 0},
        "ksei":   {"found":  3, "uploaded": 2, "skipped": 0, "failed": 1},
        "emiten": {"found":  2, "uploaded": 2, "skipped": 0, "failed": 0},
    }
    print_summary(stats)
    captured = capsys.readouterr()
    assert "IDX" in captured.out
    assert "OJK" in captured.out
    assert "KSEI" in captured.out
    assert "Emiten" in captured.out
    assert "20" in captured.out  # total = 20


def test_build_pdf_pipeline_returns_skip_for_seen():
    mock_tracker = MagicMock()
    mock_tracker.is_seen.return_value = True
    mock_uploader = MagicMock()

    item = {
        "url": "http://dup.com/file.pdf",
        "filename": "20240101_ABCD_file.pdf",
        "emiten_code": "ABCD",
        "source": "idx",
    }
    result = build_pdf_pipeline(item, mock_tracker, mock_uploader)
    assert result == "skip"


def test_build_pdf_pipeline_returns_process_for_new():
    mock_tracker = MagicMock()
    mock_tracker.is_seen.return_value = False
    mock_uploader = MagicMock()

    item = {
        "url": "http://new.com/file.pdf",
        "filename": "20240101_ABCD_file.pdf",
        "emiten_code": "ABCD",
        "source": "idx",
    }
    result = build_pdf_pipeline(item, mock_tracker, mock_uploader)
    assert result == "process"


def test_print_summary_calculates_total_correctly(capsys):
    stats = {
        "idx": {"found": 5, "uploaded": 4, "skipped": 1, "failed": 0},
        "ojk": {"found": 3, "uploaded": 3, "skipped": 0, "failed": 0},
    }
    print_summary(stats)
    captured = capsys.readouterr()
    # Total found = 8
    assert "8" in captured.out
