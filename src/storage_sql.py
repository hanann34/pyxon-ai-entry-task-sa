"""SQLite storage for document + chunk metadata"""

import sqlite3
from datetime import datetime
from typing import List, Tuple

DB_PATH = "data.sqlite3"


def get_conn() -> sqlite3.Connection:
    """Open a SQLite connection ,creates DB file if missing."""
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    """Create tables if they don't exist."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            filetype TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_uid TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            strategy TEXT NOT NULL,
            has_diacritics INTEGER NOT NULL,
            char_count INTEGER NOT NULL,
            preview TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
        """
    )

    conn.commit()
    conn.close()


def upsert_document(doc_id: str, filename: str, filetype: str) -> None:
    """Insert or replace a document row."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO documents (doc_id, filename, filetype, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (doc_id, filename, filetype, datetime.utcnow().isoformat()),
    )

    conn.commit()
    conn.close()


def upsert_chunk(
    chunk_uid: str,
    doc_id: str,
    chunk_index: int,
    strategy: str,
    has_diacritics: bool,
    char_count: int,
    preview: str,
) -> None:
    """Insert or replace one chunk metadata row."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT OR REPLACE INTO chunks
        (chunk_uid, doc_id, chunk_index, strategy, has_diacritics, char_count, preview, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            chunk_uid,
            doc_id,
            chunk_index,
            strategy,
            1 if has_diacritics else 0,
            char_count,
            preview,
            datetime.utcnow().isoformat(),
        ),
    )

    conn.commit()
    conn.close()


def list_docs(limit: int = 20) -> List[Tuple]:
    """List recent documents (small helper for debugging)."""
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT doc_id, filename, filetype, created_at
        FROM documents
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cur.fetchall()
    conn.close()
    return rows
