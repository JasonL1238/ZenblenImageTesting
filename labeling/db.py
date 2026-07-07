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

# --- multi-mode labeler (app_multi.py) --------------------------------------
# A SECOND, self-contained pipeline that shares this DB's image registry but
# writes ONLY to the additive `annotations` / `mode_status` tables below — the
# `labels` table above (the container/standard container-detection dataset) is
# never touched. Each mode is an INDEPENDENT segmentation pass over the same
# image pool and exports to its OWN single-class dataset (see export_multi.py):
#   standard -> smoothie inside the cup   -> class 0: smoothie
#   spill    -> smoothie outside the cup  -> class 0: spill
#   logo     -> the zenblen logo/wordmark -> class 0: logo
# One image labeled in all three modes yields three SEPARATE image+label pairs,
# one per dataset — never a single file with mixed-class labels.
MODES = ("standard", "spill", "logo")
MODE_CLASS_NAMES = {"standard": "smoothie", "spill": "spill", "logo": "logo"}
# Persisted statuses. "skip" is deliberately NOT stored (Skip = advance without
# writing state, so the image stays undecided and reappears later).
MODE_STATUSES = ("labeled", "clean")


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

        -- Multi-mode labeler (app_multi.py). Additive; the tables above are
        -- untouched. `annotations` is multi-instance: one row per polygon, so
        -- an image can hold several same-class shapes for a given mode.
        CREATE TABLE IF NOT EXISTS annotations (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id    INTEGER NOT NULL REFERENCES files(file_id),
            mode       TEXT NOT NULL,      -- one of db.MODES
            polygon    TEXT NOT NULL,      -- JSON list of [x, y] pixel points
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_annotations_file_mode
            ON annotations(file_id, mode);

        -- One decision per (image, mode). status is one of db.MODE_STATUSES.
        -- No row for a (file, mode) means "undecided" -> served again as next.
        CREATE TABLE IF NOT EXISTS mode_status (
            file_id    INTEGER NOT NULL REFERENCES files(file_id),
            mode       TEXT NOT NULL,
            status     TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (file_id, mode)
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
