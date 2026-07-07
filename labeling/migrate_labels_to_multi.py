"""One-time merge: fold the OLD container tool's `labels` verdicts into the new
multi-mode labeler as `mode='standard'` annotations.

After this runs, the new tool (app_multi.py) is the single labeling pipeline and
its standard mode shows/edits your previous work. The old `labels` table is left
untouched as a frozen backup — it is no longer the source of truth.

Verdict -> new state:
  corrected -> labeled, polygon = your edited polygon (SQLite), else SAM candidate
  good      -> labeled, polygon = SAM candidate (data/polygons_sam/<id>.json)
  bad/skip  -> NOT migrated -> stays undecided -> resurfaces in the new tool to redraw

Idempotent: by default skips any file that already has a standard decision in the
new tool (so re-running never clobbers edits made in app_multi). Use --force to
re-import from `labels` regardless.

Run:
  python labeling/migrate_labels_to_multi.py           # migrate
  python labeling/migrate_labels_to_multi.py --dry-run  # report only
  python labeling/migrate_labels_to_multi.py --force    # re-import, overwrite
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labeling import db

MODE = "standard"


def _source_polygon(fid: int, verdict: str, polygon_json: str | None) -> list | None:
    """Points for this label, or None if no usable polygon (>=3 pts) exists."""
    if verdict == "corrected" and polygon_json:
        pts = json.loads(polygon_json)
        if len(pts) >= 3:
            return pts
    sam_path = db.POLYGONS_DIR / f"{fid}.json"
    if sam_path.exists():
        pts = json.loads(sam_path.read_text()).get("points", [])
        if len(pts) >= 3:
            return pts
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge old labels into the multi-mode tool")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-import even if a standard decision already exists")
    args = ap.parse_args()

    conn = db.connect()
    already = {
        r["file_id"]
        for r in conn.execute(
            "SELECT file_id FROM mode_status WHERE mode = ?", (MODE,)
        )
    }
    rows = conn.execute(
        "SELECT file_id, verdict, polygon FROM labels "
        "WHERE verdict IN ('good','corrected') ORDER BY file_id"
    ).fetchall()

    migrated = skipped_existing = skipped_noimg = skipped_nopoly = 0
    for r in rows:
        fid = r["file_id"]
        if fid in already and not args.force:
            skipped_existing += 1
            continue
        if not (db.IMAGES_DIR / f"{fid}.jpg").exists():
            skipped_noimg += 1
            continue
        pts = _source_polygon(fid, r["verdict"], r["polygon"])
        if pts is None:
            skipped_nopoly += 1
            continue
        if not args.dry_run:
            conn.execute("DELETE FROM annotations WHERE file_id = ? AND mode = ?",
                         (fid, MODE))
            conn.execute(
                "INSERT INTO annotations (file_id, mode, polygon, created_at) "
                "VALUES (?, ?, ?, datetime('now'))",
                (fid, MODE, json.dumps([[int(x), int(y)] for x, y in pts])),
            )
            conn.execute(
                "INSERT INTO mode_status (file_id, mode, status, updated_at) "
                "VALUES (?, ?, 'labeled', datetime('now')) "
                "ON CONFLICT(file_id, mode) DO UPDATE SET "
                "status='labeled', updated_at=datetime('now')",
                (fid, MODE),
            )
        migrated += 1

    if not args.dry_run:
        conn.commit()

    tag = "[dry-run] would migrate" if args.dry_run else "migrated"
    print(f"{tag} {migrated} labels -> mode='standard'")
    if skipped_existing:
        print(f"  skipped {skipped_existing} (already have a standard decision; --force to overwrite)")
    if skipped_noimg:
        print(f"  skipped {skipped_noimg} (image not on disk)")
    if skipped_nopoly:
        print(f"  skipped {skipped_nopoly} (no usable polygon >=3 pts)")
    print("  (bad/skip verdicts intentionally not migrated -> redraw in the new tool)")
    print("  old `labels` table left untouched as a frozen backup")


if __name__ == "__main__":
    main()
