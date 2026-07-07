"""Stage 3 — Flask labeling UI.

Serves a keyboard-driven page that shows each image with SAM's candidate polygon
overlaid. You accept (good), reject (bad), skip, or drag polygon vertices to
correct a near-miss (auto-marked 'corrected'). Verdicts persist to SQLite so
labeling is resumable across sessions.

Run (no torch needed here — SAM already ran in stage 2):
  python labeling/app.py            # http://127.0.0.1:5000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from labeling import db

app = Flask(__name__, template_folder=str(db.ROOT / "templates"),
            static_folder=str(db.ROOT / "static"))


def _load_sam_polygon(file_id: int) -> dict | None:
    p = db.POLYGONS_DIR / f"{file_id}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


@app.route("/")
def index() -> str:
    return send_from_directory(str(db.ROOT / "templates"), "label.html")


@app.route("/image/<int:file_id>")
def image(file_id: int):
    path = db.IMAGES_DIR / f"{file_id}.jpg"
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype="image/jpeg")


@app.route("/mask/<int:file_id>")
def mask(file_id: int):
    path = db.MASKS_DIR / f"{file_id}.png"
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype="image/png")


@app.route("/api/next")
def api_next():
    """Return the next item needing a label.

    Only considers files that were downloaded AND have a SAM polygon on disk.
    ``?after=<id>`` returns the first such item with a larger file_id (used to
    step forward); otherwise returns the first unlabeled item.
    """
    after = request.args.get("after", type=int)
    conn = db.connect()
    base = (
        "SELECT f.file_id FROM files f "
        "LEFT JOIN labels l ON l.file_id = f.file_id "
        "WHERE f.downloaded = 1 AND l.file_id IS NULL "
    )
    if after is not None:
        row = conn.execute(
            base + "AND f.file_id > ? ORDER BY f.file_id ASC LIMIT 1", (after,)
        ).fetchone()
    else:
        row = conn.execute(base + "ORDER BY f.file_id ASC LIMIT 1").fetchone()

    # Skip rows whose SAM polygon isn't on disk yet (stage 2 not run for them).
    while row is not None:
        poly = _load_sam_polygon(row["file_id"])
        if poly is not None:
            return jsonify({
                "file_id": row["file_id"],
                "width": poly["width"],
                "height": poly["height"],
                "polygon": poly["points"],
            })
        row = conn.execute(
            base + "AND f.file_id > ? ORDER BY f.file_id ASC LIMIT 1",
            (row["file_id"],),
        ).fetchone()

    return jsonify({"done": True})


@app.route("/api/item/<int:file_id>")
def api_item(file_id: int):
    """Load a specific item, preferring a previously-saved (corrected) polygon."""
    conn = db.connect()
    frow = conn.execute("SELECT file_id FROM files WHERE file_id = ?", (file_id,)).fetchone()
    if frow is None:
        abort(404)
    sam = _load_sam_polygon(file_id)
    if sam is None:
        abort(404)
    lrow = conn.execute("SELECT verdict, polygon FROM labels WHERE file_id = ?",
                        (file_id,)).fetchone()
    points = sam["points"]
    verdict = None
    if lrow is not None:
        verdict = lrow["verdict"]
        if lrow["polygon"]:
            points = json.loads(lrow["polygon"])
    return jsonify({
        "file_id": file_id,
        "width": sam["width"],
        "height": sam["height"],
        "polygon": points,
        "verdict": verdict,
    })


@app.route("/api/label", methods=["POST"])
def api_label():
    data = request.get_json(force=True)
    file_id = data.get("file_id")
    verdict = data.get("verdict")
    polygon = data.get("polygon")
    if file_id is None or verdict not in db.VERDICTS:
        abort(400, f"file_id required and verdict must be one of {db.VERDICTS}")
    conn = db.connect()
    poly_json = json.dumps(polygon) if polygon else None
    conn.execute(
        """
        INSERT INTO labels (file_id, verdict, polygon, labeled_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(file_id) DO UPDATE SET
            verdict=excluded.verdict, polygon=excluded.polygon,
            labeled_at=excluded.labeled_at
        """,
        (file_id, verdict, poly_json),
    )
    conn.commit()
    return jsonify({"ok": True})


@app.route("/api/progress")
def api_progress():
    conn = db.connect()
    total = conn.execute("SELECT COUNT(*) c FROM files WHERE downloaded = 1").fetchone()["c"]
    labeled = conn.execute("SELECT COUNT(*) c FROM labels").fetchone()["c"]
    counts = {v: 0 for v in db.VERDICTS}
    for r in conn.execute("SELECT verdict, COUNT(*) c FROM labels GROUP BY verdict"):
        counts[r["verdict"]] = r["c"]
    return jsonify({"total": total, "labeled": labeled, **counts})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the labeling web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print("\n  [DEPRECATED] This is the OLD single-mode container labeler.\n"
          "  Labeling is now unified in app_multi.py (standard/spill/logo).\n"
          "  Its `labels` writes here are NO LONGER the source of truth — the\n"
          "  new tool reads migrated annotations from the `annotations` table.\n"
          "  Use:  python labeling/app_multi.py   (http://127.0.0.1:5001)\n")
    db.ensure_dirs()
    db.connect().close()  # create schema up front
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
