"""Shared SQLite helpers for the labeling tool.

One database (``labeling/labels.db``) holds two tables:

  files   — one row per downloaded image (populated by ``download.py``)
  labels  — one row per human verdict (populated by ``app.py``)

Both are keyed by ``file_id`` (the Files-API file id, an integer PK), so the
download / SAM / label / export stages all join cleanly on that id.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

# All labeling artefacts live under this package directory, next to the scripts.
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "labels.db"
IMAGES_DIR = ROOT / "data" / "images"
MASKS_DIR = ROOT / "data" / "masks_sam"
POLYGONS_DIR = ROOT / "data" / "polygons_sam"
DATASET_DIR = ROOT / "dataset"

VERDICTS = ("good", "corrected", "bad", "skip")


def connect(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Open the labeling DB, creating the schema on first use.

    ``db_path`` resolves to the module-level ``DB_PATH`` at call time (not bound
    as a default), so tests can point it elsewhere by reassigning ``db.DB_PATH``.
    """
    conn = sqlite3.connect(str(db_path if db_path is not None else DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            file_id       INTEGER PRIMARY KEY,
            order_id      INTEGER,
            file_name     TEXT,
            file_url      TEXT,
            category_name TEXT,
            file_type     TEXT,
            created_at    TEXT,
            downloaded    INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS labels (
            file_id    INTEGER PRIMARY KEY REFERENCES files(file_id),
            verdict    TEXT NOT NULL,
            polygon    TEXT,               -- JSON list of [x, y] pixel points
            labeled_at TEXT NOT NULL
        );
        """
    )
    conn.commit()


def load_dotenv() -> None:
    """Load ``KEY=VALUE`` lines from a ``.env`` file into ``os.environ``.

    Dependency-free (no python-dotenv). Looks for ``labeling/.env`` first, then
    the repo-root ``.env``. Existing environment variables win, so an explicit
    ``export`` or ``--api-key`` still overrides the file. Missing file is fine.
    """
    for env_path in (ROOT / ".env", ROOT.parent / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def ensure_dirs() -> None:
    """Create the on-disk data directories used by the pipeline stages."""
    for d in (IMAGES_DIR, MASKS_DIR, POLYGONS_DIR):
        d.mkdir(parents=True, exist_ok=True)
