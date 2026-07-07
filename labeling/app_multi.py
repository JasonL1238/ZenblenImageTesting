"""Multi-mode segmentation labeling UI (spill / logo / standard).

A SECOND labeling app that runs ALONGSIDE the existing container tool
(``app.py``) without touching its data. Same image pool (the ``files`` table in
``labels.db``), but every write goes to the additive ``annotations`` /
``mode_status`` tables — the ``labels`` table and ``smoothie_dataset`` are never
modified here.

Three INDEPENDENT modes, each its own single-class YOLO-seg dataset:
  1 standard — smoothie inside the cup   (class 0: smoothie)
  2 spill    — smoothie outside the cup  (class 0: spill)   [new YOLO-nano]
  3 logo     — the zenblen logo/wordmark (class 0: logo)    [new YOLO-nano]

Switching mode keeps the CURRENT image and loads that image's annotations for
the new mode, so one source image can be segmented separately in each mode.
Export per mode with ``export_multi.py``.

Run (no torch needed — free-draw, SAM only seeds standard mode if present):
  python labeling/app_multi.py            # http://127.0.0.1:5001
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


def _check_mode(mode: str | None) -> str:
    if mode not in db.MODES:
        abort(400, f"mode must be one of {db.MODES}")
    return mode


@app.route("/")
def index() -> str:
    return send_from_directory(str(db.ROOT / "templates"), "label_multi.html")


@app.route("/image/<int:file_id>")
def image(file_id: int):
    path = db.IMAGES_DIR / f"{file_id}.jpg"
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype="image/jpeg")


@app.route("/api/sam/<int:file_id>")
def api_sam(file_id: int):
    """Optional SAM candidate polygon (standard mode seed). {} if none on disk."""
    p = db.POLYGONS_DIR / f"{file_id}.json"
    if not p.exists():
        return jsonify({})
    data = json.loads(p.read_text())
    return jsonify({"polygon": data.get("points", [])})


def _shapes_for(conn, file_id: int, mode: str) -> list[list[list[int]]]:
    rows = conn.execute(
        "SELECT polygon FROM annotations WHERE file_id = ? AND mode = ? ORDER BY id",
        (file_id, mode),
    ).fetchall()
    return [json.loads(r["polygon"]) for r in rows]


@app.route("/api/next")
def api_next():
    """First downloaded image with NO decision yet in ``mode``.

    Unlike the container tool this does NOT require a SAM polygon on disk —
    spill/logo are free-drawn. ``?after=<id>`` steps forward past a given id.
    """
    mode = _check_mode(request.args.get("mode"))
    after = request.args.get("after", type=int)
    conn = db.connect()
    base = (
        "SELECT f.file_id FROM files f "
        "LEFT JOIN mode_status m ON m.file_id = f.file_id AND m.mode = ? "
        "WHERE f.downloaded = 1 AND m.file_id IS NULL "
    )
    params: tuple = (mode,)
    if after is not None:
        row = conn.execute(
            base + "AND f.file_id > ? ORDER BY f.file_id ASC LIMIT 1",
            params + (after,),
        ).fetchone()
    else:
        row = conn.execute(
            base + "ORDER BY f.file_id ASC LIMIT 1", params
        ).fetchone()

    # Only serve images whose jpg is actually on disk.
    while row is not None:
        if (db.IMAGES_DIR / f"{row['file_id']}.jpg").exists():
            return jsonify({"file_id": row["file_id"], "mode": mode,
                            "status": None, "shapes": []})
        row = conn.execute(
            base + "AND f.file_id > ? ORDER BY f.file_id ASC LIMIT 1",
            params + (row["file_id"],),
        ).fetchone()
    return jsonify({"done": True})


@app.route("/api/item/<int:file_id>")
def api_item(file_id: int):
    """Load a specific image's annotations for ``mode`` (image dims come from the
    loaded <img> client-side, so none are returned here)."""
    mode = _check_mode(request.args.get("mode"))
    conn = db.connect()
    if conn.execute("SELECT 1 FROM files WHERE file_id = ?", (file_id,)).fetchone() is None:
        abort(404)
    srow = conn.execute(
        "SELECT status FROM mode_status WHERE file_id = ? AND mode = ?",
        (file_id, mode),
    ).fetchone()
    return jsonify({
        "file_id": file_id,
        "mode": mode,
        "status": srow["status"] if srow else None,
        "shapes": _shapes_for(conn, file_id, mode),
    })


@app.route("/api/annotate", methods=["POST"])
def api_annotate():
    """Persist a decision for (file_id, mode).

    Body: {file_id, mode, status in db.MODE_STATUSES, shapes:[{polygon:[[x,y]..]}]}
      labeled -> replace this mode's annotations with the given shapes (>=3 pts)
      clean   -> no shapes; exports as an empty (background) label file
    Skip is client-only (advance without writing), so it never reaches here.
    """
    data = request.get_json(force=True)
    file_id = data.get("file_id")
    mode = _check_mode(data.get("mode"))
    status = data.get("status")
    if file_id is None or status not in db.MODE_STATUSES:
        abort(400, f"file_id required and status must be one of {db.MODE_STATUSES}")

    shapes = data.get("shapes") or []
    polys: list[list[list[int]]] = []
    if status == "labeled":
        for s in shapes:
            pts = s.get("polygon") if isinstance(s, dict) else s
            if pts and len(pts) >= 3:
                polys.append([[int(x), int(y)] for x, y in pts])
        if not polys:
            abort(400, "labeled requires at least one polygon with >=3 points")

    conn = db.connect()
    # Replace this (file, mode)'s annotations wholesale — strictly scoped by
    # `mode`, so other modes' rows are never affected (zero cross-contamination).
    conn.execute("DELETE FROM annotations WHERE file_id = ? AND mode = ?",
                 (file_id, mode))
    for pts in polys:
        conn.execute(
            "INSERT INTO annotations (file_id, mode, polygon, created_at) "
            "VALUES (?, ?, ?, datetime('now'))",
            (file_id, mode, json.dumps(pts)),
        )
    conn.execute(
        """
        INSERT INTO mode_status (file_id, mode, status, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(file_id, mode) DO UPDATE SET
            status=excluded.status, updated_at=excluded.updated_at
        """,
        (file_id, mode, status),
    )
    conn.commit()
    return jsonify({"ok": True})


@app.route("/api/progress")
def api_progress():
    """Per-mode counts. ``?mode=`` restricts to one mode; otherwise all modes."""
    conn = db.connect()
    total = conn.execute(
        "SELECT COUNT(*) c FROM files WHERE downloaded = 1"
    ).fetchone()["c"]
    modes = [request.args["mode"]] if request.args.get("mode") else list(db.MODES)
    out: dict = {"total": total, "modes": {}}
    for m in modes:
        _check_mode(m)
        counts = {s: 0 for s in db.MODE_STATUSES}
        for r in conn.execute(
            "SELECT status, COUNT(*) c FROM mode_status WHERE mode = ? GROUP BY status",
            (m,),
        ):
            counts[r["status"]] = r["c"]
        counts["decided"] = sum(counts[s] for s in db.MODE_STATUSES)
        out["modes"][m] = counts
    return jsonify(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the multi-mode labeling UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    db.ensure_dirs()
    db.connect().close()  # create schema (incl. new tables) up front
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
