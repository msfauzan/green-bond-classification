import os
import tempfile
import pytest
from prospektus_collector.tracker import Tracker


@pytest.fixture
def tracker():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    t = Tracker(db_path=db_path)
    yield t
    # Windows may hold SQLite file lock briefly; ignore cleanup errors
    try:
        os.unlink(db_path)
    except OSError:
        pass


def test_tracker_is_not_seen_initially(tracker):
    assert tracker.is_seen("http://example.com/file.pdf") is False


def test_tracker_mark_and_check(tracker):
    tracker.mark(
        url="http://example.com/file.pdf",
        filename="20240101_ABCD_file.pdf",
        source="idx",
        emiten_code="ABCD",
        r2_key="prospektus/idx/ABCD/20240101_ABCD_file.pdf",
        status="uploaded",
    )
    assert tracker.is_seen("http://example.com/file.pdf") is True


def test_tracker_failed_not_seen(tracker):
    tracker.mark(
        url="http://example.com/fail.pdf",
        filename="20240101_ABCD_fail.pdf",
        source="ojk",
        emiten_code="ABCD",
        r2_key="",
        status="failed",
    )
    assert tracker.is_seen("http://example.com/fail.pdf") is False


def test_tracker_get_failed(tracker):
    tracker.mark(
        url="http://example.com/fail.pdf",
        filename="20240101_ABCD_fail.pdf",
        source="ojk",
        emiten_code="ABCD",
        r2_key="",
        status="failed",
    )
    failed = tracker.get_failed()
    assert len(failed) == 1
    assert failed[0]["url"] == "http://example.com/fail.pdf"


def test_tracker_summary(tracker):
    tracker.mark("http://a.com/1.pdf", "f1.pdf", "idx", "AAAA", "k1", "uploaded")
    tracker.mark("http://a.com/2.pdf", "f2.pdf", "ojk", "BBBB", "k2", "uploaded")
    tracker.mark("http://a.com/3.pdf", "f3.pdf", "idx", "CCCC", "", "failed")
    summary = tracker.summary()
    assert summary["total"] == 3
    assert summary["uploaded"] == 2
    assert summary["failed"] == 1


def test_tracker_update_status(tracker):
    tracker.mark("http://a.com/1.pdf", "f1.pdf", "idx", "AAAA", "", "failed")
    tracker.update_status("http://a.com/1.pdf", "uploaded", r2_key="k1")
    assert tracker.is_seen("http://a.com/1.pdf") is True
    failed = tracker.get_failed()
    assert len(failed) == 0


def test_tracker_summary_by_source(tracker):
    tracker.mark("http://a.com/1.pdf", "f1.pdf", "idx", "AAAA", "k1", "uploaded")
    tracker.mark("http://a.com/2.pdf", "f2.pdf", "idx", "BBBB", "k2", "uploaded")
    tracker.mark("http://a.com/3.pdf", "f3.pdf", "ojk", "CCCC", "", "failed")
    by_source = tracker.summary_by_source()
    assert "idx" in by_source
    assert by_source["idx"]["uploaded"] == 2
    assert by_source["ojk"]["failed"] == 1
