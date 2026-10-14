import sqlite3
import json
from pathlib import Path
import hashlib

class RAGCache:
    def __init__(self, db_path="query_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Ensures the cache table exists with support for query_hash or query."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    query_hash TEXT PRIMARY KEY,
                    query TEXT,
                    answer TEXT,
                    sources TEXT
                )
            """)
            conn.commit()

    def _get_hash(self, query: str) -> str:
        return hashlib.sha256(query.strip().lower().encode()).hexdigest()

    def get(self, query: str):
        self._init_db()
        q_key = self._get_hash(query)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Try querying by query_hash first, fallback to query text if schema differs
            try:
                cursor.execute("SELECT answer, sources FROM cache WHERE query_hash = ?", (q_key,))
            except sqlite3.OperationalError:
                cursor.execute("SELECT answer, sources FROM cache WHERE query = ?", (query,))
                
            row = cursor.fetchone()
            if row:
                return {
                    "answer": row[0],
                    "sources": json.loads(row[1]) if row[1] else []
                }
        return None

    def set(self, query: str, answer, sources: list):
        self._init_db()
        q_key = self._get_hash(query)
        with sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (query_hash, query, answer, sources) VALUES (?, ?, ?, ?)",
                    (q_key, query, str(answer), json.dumps(sources))
                )
            except sqlite3.OperationalError:
                # Fallback for old table structure
                conn.execute(
                    "INSERT OR REPLACE INTO cache (query, answer, sources) VALUES (?, ?, ?)",
                    (query, str(answer), json.dumps(sources))
                )
            conn.commit()

    def clear(self):
        self._init_db()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()