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


def _item_payload(conn, file_id: int, mode: str) -> dict:
    """This mode's saved decision + shapes for one image (the client-facing item).

    ``priority`` marks whether this image is listed in ``priority/<mode>.txt`` for
    the active mode (independent per mode), so the UI can badge front-of-queue
    images — true regardless of whether it's still undecided or already labeled.
    """
    srow = conn.execute(
        "SELECT status FROM mode_status WHERE file_id = ? AND mode = ?",
        (file_id, mode),
    ).fetchone()
    return {
        "file_id": file_id,
        "mode": mode,
        "status": srow["status"] if srow else None,
        "shapes": _shapes_for(conn, file_id, mode),
        "priority": file_id in set(_priority_ids(mode)),
    }


def _priority_ids(mode: str) -> list[int]:
    """Optional front-of-queue file_ids from ``labeling/priority/<mode>.txt``.

    One integer id per line (# comments / blanks ignored). Used so hard cases
    (e.g. spill reflection / tiny-droplet lookalikes) get labeled before the
    default ascending file_id walk.
    """
    path = db.ROOT / "priority" / f"{mode}.txt"
    if not path.exists():
        return []
    out: list[int] = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            out.append(int(line))
        except ValueError:
            continue
    return out


@app.route("/api/next")
def api_next():
    """First downloaded image with NO decision yet in ``mode``.

    Unlike the container tool this does NOT require a SAM polygon on disk —
    spill/logo are free-drawn. ``?after=<id>`` steps forward past a given id.

    If ``labeling/priority/<mode>.txt`` exists, undecided ids listed there are
    served first (in file order), then the normal ascending file_id walk.
    """
    mode = _check_mode(request.args.get("mode"))
    after = request.args.get("after", type=int)
    conn = db.connect()

    # Priority queue: serve undecided ids from labeling/priority/<mode>.txt
    # first. When ?after= points at a priority id, continue AFTER that entry so
    # Skip still advances; once the remaining priority list is exhausted, fall
    # through to the normal ascending walk.
    priority = _priority_ids(mode)
    prio_set = set(priority)
    if priority:
        undecided = {
            r["file_id"]
            for r in conn.execute(
                "SELECT f.file_id FROM files f "
                "LEFT JOIN mode_status m ON m.file_id = f.file_id AND m.mode = ? "
                "LEFT JOIN image_flags fl ON fl.file_id = f.file_id "
                "LEFT JOIN review_status rs ON rs.file_id = f.file_id AND rs.mode = ? "
                "WHERE f.downloaded = 1 AND m.file_id IS NULL "
                "  AND fl.file_id IS NULL "                       # skip machinery
                "  AND (rs.status IS NULL OR rs.status != 'pending')",  # not still in auto-review
                (mode, mode),
            )
        }
        start = 0
        if after is not None:
            try:
                start = priority.index(after) + 1
            except ValueError:
                start = 0
        for fid in priority[start:]:
            if fid in undecided and (db.IMAGES_DIR / f"{fid}.jpg").exists():
                # preload any kept polygons (e.g. relabel pass: old wordmark shape
                # shows so you edit/add rather than redraw). Blank images -> [].
                return jsonify({"file_id": fid, "mode": mode,
                                "status": None, "shapes": _shapes_for(conn, fid, mode),
                                "priority": True})

    # Serve only images that BELONG in manual: undecided, not machinery, and NOT
    # still pending in auto-review (those live in app_review.py until you reject
    # them here). Rejected images (review_status='rejected', no mode_status) and
    # never-predicted images pass; pending ones are held back so the two queues
    # never overlap and rejects flow in live as you review.
    base = (
        "SELECT f.file_id FROM files f "
        "LEFT JOIN mode_status m ON m.file_id = f.file_id AND m.mode = ? "
        "LEFT JOIN image_flags fl ON fl.file_id = f.file_id "
        "LEFT JOIN review_status rs ON rs.file_id = f.file_id AND rs.mode = ? "
        "WHERE f.downloaded = 1 AND m.file_id IS NULL AND fl.file_id IS NULL "
        "  AND (rs.status IS NULL OR rs.status != 'pending') "
    )
    params: tuple = (mode, mode)
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
                            "status": None,
                            "shapes": _shapes_for(conn, row["file_id"], mode),
                            "priority": row["file_id"] in prio_set})
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
    return jsonify(_item_payload(conn, file_id, mode))


@app.route("/api/seek")
def api_seek():
    """Adjacent downloaded image by ``file_id`` in ``dir`` (prev|next), REGARDLESS
    of whether it already has a decision in ``mode``.

    Unlike ``/api/next`` (which only serves undecided images), this walks the whole
    on-disk pool, so you can always go back and re-edit a previously-labeled image —
    including one decided on an earlier run. Returns the same payload as
    ``/api/item``, or ``{"edge": True}`` when there is no further image that way.
    """
    mode = _check_mode(request.args.get("mode"))
    file_id = request.args.get("file_id", type=int)
    direction = request.args.get("dir", "prev")
    if file_id is None or direction not in ("prev", "next"):
        abort(400, "file_id required and dir must be 'prev' or 'next'")
    op, order = ("<", "DESC") if direction == "prev" else (">", "ASC")
    sql = (
        f"SELECT file_id FROM files WHERE downloaded = 1 AND file_id {op} ? "
        f"ORDER BY file_id {order} LIMIT 1"
    )
    conn = db.connect()
    cur = file_id
    # Skip past any pool entry whose jpg is not actually on disk (mirrors /api/next).
    while True:
        row = conn.execute(sql, (cur,)).fetchone()
        if row is None:
            return jsonify({"edge": True})
        if (db.IMAGES_DIR / f"{row['file_id']}.jpg").exists():
            return jsonify(_item_payload(conn, row["file_id"], mode))
        cur = row["file_id"]


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


def _undecided_ids(conn, mode: str) -> set[int]:
    """file_ids the hand labeler would still serve for ``mode`` — undecided, not
    machinery, not still pending in auto-review. Mirrors ``/api/next``'s gate so
    the priority count matches what actually gets served."""
    return {
        r["file_id"]
        for r in conn.execute(
            "SELECT f.file_id FROM files f "
            "LEFT JOIN mode_status m ON m.file_id = f.file_id AND m.mode = ? "
            "LEFT JOIN image_flags fl ON fl.file_id = f.file_id "
            "LEFT JOIN review_status rs ON rs.file_id = f.file_id AND rs.mode = ? "
            "WHERE f.downloaded = 1 AND m.file_id IS NULL AND fl.file_id IS NULL "
            "  AND (rs.status IS NULL OR rs.status != 'pending')",
            (mode, mode),
        )
    }


@app.route("/api/progress")
def api_progress():
    """Per-mode counts. ``?mode=`` restricts to one mode; otherwise all modes.

    Each mode also reports its OWN priority-queue counts (from
    ``labeling/priority/<mode>.txt``), independent of every other mode:
      ``priority_total``     — ids listed in that mode's file
      ``priority_remaining`` — of those, how many are still undecided (i.e. left
                               to label; excludes machinery / pending-review ids).
    """
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
        priority = _priority_ids(m)
        counts["priority_total"] = len(priority)
        if priority:
            undecided = _undecided_ids(conn, m)
            counts["priority_remaining"] = sum(1 for fid in priority if fid in undecided)
        else:
            counts["priority_remaining"] = 0
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
