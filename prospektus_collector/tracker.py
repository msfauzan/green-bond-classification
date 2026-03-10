import sqlite3
from datetime import datetime, timezone
from typing import Optional


class Tracker:
    """SQLite-backed deduplication tracker untuk PDF prospektus."""

    def __init__(self, db_path: str = "collector_tracker.db"):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS downloads (
                    url TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source TEXT NOT NULL,
                    emiten_code TEXT NOT NULL,
                    r2_key TEXT,
                    status TEXT NOT NULL,
                    crawled_at TEXT NOT NULL
                )
            """)

    def is_seen(self, url: str) -> bool:
        """Return True hanya jika PDF sudah berhasil di-upload."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT 1 FROM downloads WHERE url = ? AND status = 'uploaded'",
                (url,),
            ).fetchone()
            return row is not None

    def mark(
        self,
        url: str,
        filename: str,
        source: str,
        emiten_code: str,
        r2_key: str,
        status: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO downloads
                    (url, filename, source, emiten_code, r2_key, status, crawled_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (url, filename, source, emiten_code, r2_key, status, datetime.now(timezone.utc).isoformat()),
            )

    def get_failed(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM downloads WHERE status = 'failed'"
            ).fetchall()
            return [dict(r) for r in rows]

    def update_status(self, url: str, status: str, r2_key: Optional[str] = None) -> None:
        with self._conn() as conn:
            if r2_key is not None:
                conn.execute(
                    "UPDATE downloads SET status = ?, r2_key = ? WHERE url = ?",
                    (status, r2_key, url),
                )
            else:
                conn.execute(
                    "UPDATE downloads SET status = ? WHERE url = ?",
                    (status, url),
                )

    def summary(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM downloads").fetchone()[0]
            uploaded = conn.execute(
                "SELECT COUNT(*) FROM downloads WHERE status = 'uploaded'"
            ).fetchone()[0]
            failed = conn.execute(
                "SELECT COUNT(*) FROM downloads WHERE status = 'failed'"
            ).fetchone()[0]
            skipped = conn.execute(
                "SELECT COUNT(*) FROM downloads WHERE status = 'skipped'"
            ).fetchone()[0]
        return {"total": total, "uploaded": uploaded, "failed": failed, "skipped": skipped}

    def summary_by_source(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT source,
                       COUNT(*) as total,
                       SUM(CASE WHEN status='uploaded' THEN 1 ELSE 0 END) as uploaded,
                       SUM(CASE WHEN status='skipped' THEN 1 ELSE 0 END) as skipped,
                       SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed
                FROM downloads
                GROUP BY source
            """).fetchall()
        return {row["source"]: dict(row) for row in rows}
