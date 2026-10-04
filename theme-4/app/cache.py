import hashlib
import json
import sqlite3
from typing import Optional

DB_PATH = "query_cache.db"


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            query_hash TEXT PRIMARY KEY,
            query_text TEXT,
            result_json TEXT
        )
    """)
    conn.commit()
    conn.close()


_init_db()


def _normalize(query: str) -> str:
    # Lowercased, whitespace-collapsed - so trivial rephrasing/casing still hits
    # the same cache entry (e.g. "What's the capacity?" == "whats the capacity").
    return " ".join(query.strip().lower().split())


def _hash(query: str) -> str:
    return hashlib.sha256(_normalize(query).encode("utf-8")).hexdigest()


def get_cached(query: str) -> Optional[dict]:
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT result_json FROM cache WHERE query_hash = ?", (_hash(query),)
    ).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def set_cached(query: str, result: dict) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO cache (query_hash, query_text, result_json) VALUES (?, ?, ?)",
        (_hash(query), query, json.dumps(result)),
    )
    conn.commit()
    conn.close()
