"""SQLite storage for the trial service: users + weekly usage.

MVP keeps everything in one file; swap to Postgres by replacing the
three helpers when user count warrants it.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "trial.db"


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS users (
               id INTEGER PRIMARY KEY,
               google_sub TEXT UNIQUE,
               email TEXT,
               name TEXT,
               created_at INTEGER
           )"""
    )
    c.execute(
        """CREATE TABLE IF NOT EXISTS usage (
               id INTEGER PRIMARY KEY,
               user_id INTEGER,
               week_key TEXT,        -- e.g. 2026-W47 (ISO week)
               kind TEXT,            -- trial | api
               created_at INTEGER
           )"""
    )
    c.execute("CREATE INDEX IF NOT EXISTS idx_usage ON usage(user_id, week_key)")
    return c


def upsert_user(google_sub: str, email: str, name: str) -> int:
    c = _conn()
    cur = c.execute(
        """INSERT INTO users (google_sub, email, name, created_at) VALUES (?, ?, ?, ?)
           ON CONFLICT(google_sub) DO UPDATE SET email=excluded.email, name=excluded.name""",
        (google_sub, email, name, int(time.time())),
    )
    uid = c.execute("SELECT id FROM users WHERE google_sub=?", (google_sub,)).fetchone()[0]
    c.commit()
    c.close()
    return uid


def get_user(google_sub: str) -> tuple | None:
    c = _conn()
    row = c.execute("SELECT id, email, name FROM users WHERE google_sub=?", (google_sub,)).fetchone()
    c.close()
    return row


def week_key(ts: float | None = None) -> str:
    t = time.gmtime(ts if ts is not None else time.time())
    iso = time.strftime("%G-W%V", t)
    return iso


def used_this_week(user_id: int) -> int:
    c = _conn()
    n = c.execute(
        "SELECT COUNT(*) FROM usage WHERE user_id=? AND week_key=?", (user_id, week_key())
    ).fetchone()[0]
    c.close()
    return n


def record_use(user_id: int, kind: str = "trial") -> None:
    c = _conn()
    c.execute(
        "INSERT INTO usage (user_id, week_key, kind, created_at) VALUES (?, ?, ?, ?)",
        (user_id, week_key(), kind, int(time.time())),
    )
    c.commit()
    c.close()
