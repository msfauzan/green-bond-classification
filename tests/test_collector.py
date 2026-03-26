"""Tests for prospektus collector."""
import pytest
import tempfile
import os
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from prospektus_collector.tracker import ProspektusTracker


class TestTracker:
    def test_tracker_creates_database(self):
        """Tracker should create SQLite database on init."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_tracker.db")
            tracker = ProspektusTracker(db_path)
            assert os.path.exists(db_path)

    def test_tracker_add_and_check(self):
        """Tracker should track PDF URLs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_tracker.db")
            tracker = ProspektusTracker(db_path)

            # Should not exist initially
            assert not tracker.exists("https://example.com/test.pdf")

            # Add URL
            tracker.add(
                url="https://example.com/test.pdf",
                filename="20240101_TEST_test.pdf",
                source="idx",
                emiten_code="TEST",
                r2_key="idx/TEST/20240101_TEST_test.pdf"
            )

            # Should exist now
            assert tracker.exists("https://example.com/test.pdf")

    def test_tracker_get_stats(self):
        """Tracker should return statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_tracker.db")
            tracker = ProspektusTracker(db_path)

            tracker.add("url1", "file1.pdf", "idx", "TEST1", "idx/TEST1/file1.pdf")
            tracker.add("url2", "file2.pdf", "ojk", "TEST2", "ojk/TEST2/file2.pdf")
            tracker.add("url3", "file3.pdf", "idx", "TEST3", "idx/TEST3/file3.pdf", status="failed")

            stats = tracker.get_stats()
            assert stats["total"] == 3
            assert stats["by_source"]["idx"] == 2
            assert stats["by_source"]["ojk"] == 1
            assert stats["by_status"]["completed"] == 2
            assert stats["by_status"]["failed"] == 1
