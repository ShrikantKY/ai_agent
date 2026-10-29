"""
Long-term memory that survives between runs, stored in SQLite.

Two kinds of memory:
  notes     - rules the agent learned ("ttl must be <= 2147483647"). Written by the agent.
  examples  - files that were verified to work. Written by CODE when Terraform accepts
              a file, never on the model's say-so. The model can't mark its own work as correct.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path


class Memory:
    def __init__(self, db_path: Path | str):
        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path))
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS notes (
                topic      TEXT NOT NULL,
                note       TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(topic, note)
            );
            CREATE TABLE IF NOT EXISTS examples (
                topic       TEXT NOT NULL,
                name        TEXT NOT NULL,
                content     TEXT NOT NULL,
                verified_by TEXT NOT NULL,
                created_at  REAL NOT NULL,
                UNIQUE(topic, content)
            );
        """)

    # ── Notes ────────────────────────────────────────────────────────────────
    def add_note(self, topic: str, note: str) -> bool:
        """Returns True if the note is new."""
        cur = self._db.execute("INSERT OR IGNORE INTO notes VALUES (?, ?, ?)",
                               (topic, note.strip(), time.time()))
        self._db.commit()
        return cur.rowcount == 1

    def notes(self, topic: str, limit: int = 20) -> list[str]:
        rows = self._db.execute(
            "SELECT note FROM notes WHERE topic = ? ORDER BY created_at DESC LIMIT ?", (topic, limit))
        return [r[0] for r in rows]

    # ── Verified examples ───────────────────────────────────────────────────
    def add_example(self, topic: str, name: str, content: str, verified_by: str) -> bool:
        cur = self._db.execute(
            """INSERT INTO examples VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(topic, content) DO UPDATE SET
                 verified_by = CASE WHEN excluded.verified_by = 'apply' THEN 'apply'
                                    ELSE examples.verified_by END,
                 created_at = excluded.created_at""",
            (topic, name, content, verified_by, time.time()))
        self._db.commit()
        return cur.rowcount == 1

    def examples(self, topic: str, limit: int = 2) -> list[dict]:
        rows = self._db.execute(
            """SELECT name, content, verified_by FROM examples WHERE topic = ?
               ORDER BY (verified_by = 'apply') DESC, created_at DESC LIMIT ?""", (topic, limit))
        return [{"name": n, "content": c, "verified_by": v} for n, c, v in rows]

    def close(self) -> None:
        self._db.close()
