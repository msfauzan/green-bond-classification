"""SQLite tracker for deduplication and status tracking."""
import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, Any, List


class ProspektusTracker:
    """Track collected PDFs to avoid duplicates."""

    def __init__(self, db_path: str = "prospektus_collector/tracker.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prospektus (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                source TEXT NOT NULL,
                emiten_code TEXT,
                r2_key TEXT,
                status TEXT DEFAULT 'completed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                error_message TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_url ON prospektus(url)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_emiten ON prospektus(emiten_code)
        """)

        conn.commit()
        conn.close()

    def exists(self, url: str) -> bool:
        """Check if URL has been processed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM prospektus WHERE url = ?", (url,))
        result = cursor.fetchone()
        conn.close()
        return result is not None

    def add(
        self,
        url: str,
        filename: str,
        source: str,
        emiten_code: Optional[str] = None,
        r2_key: Optional[str] = None,
        status: str = "completed",
        error_message: Optional[str] = None
    ) -> int:
        """Add a new prospektus record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO prospektus
            (url, filename, source, emiten_code, r2_key, status, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (url, filename, source, emiten_code, r2_key, status, error_message))

        record_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return record_id

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total count
        cursor.execute("SELECT COUNT(*) FROM prospektus")
        total = cursor.fetchone()[0]

        # By source
        cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM prospektus
            GROUP BY source
        """)
        by_source = {row[0]: row[1] for row in cursor.fetchall()}

        # By status
        cursor.execute("""
            SELECT status, COUNT(*) as count
            FROM prospektus
            GROUP BY status
        """)
        by_status = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            "total": total,
            "by_source": by_source,
            "by_status": by_status
        }

    def get_failed(self) -> List[Dict[str, Any]]:
        """Get all failed records for retry."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, url, filename, source, emiten_code, error_message
            FROM prospektus
            WHERE status = 'failed'
        """)

        results = cursor.fetchall()
        conn.close()

        return [
            {
                "id": r[0],
                "url": r[1],
                "filename": r[2],
                "source": r[3],
                "emiten_code": r[4],
                "error_message": r[5]
            }
            for r in results
        ]
