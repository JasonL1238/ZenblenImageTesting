"""Model-assisted review UI — approve/reject/edit YOLO predictions into training.

The SECOND stage of the review pipeline (after ``predict_batch.py`` stages
predictions). Runs ALONGSIDE the hand labeler (``app_multi.py``) on the same
image pool and DB, but is strictly separate: it reads ``predictions`` /
``review_status`` and only writes the shared training tables (``annotations`` +
``mode_status``) when a prediction is APPROVED.

Per (image, mode) decision:
  approve — save the (possibly edited) polygons to annotations with
            source='model' and mode_status='labeled'  -> enters training via
            the unchanged export_multi.py.
  clean   — no target here: mode_status='clean' (empty/negative label). Fills
            the clean-negative gap on spill/logo.
  reject  — model got it wrong: mark review_status='rejected' and leave NO
            mode_status row, so app_multi.py re-serves the image for hand
            labeling (and push it to priority/<mode>.txt so it jumps the queue).

Review order defaults to LOWEST-confidence-first (?sort=conf_asc) so effort goes
where the model is weakest; zero-detection images sort first (catch false-negs).

  python labeling/app_review.py --mode spill        # http://127.0.0.1:5002
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

_LABELING = Path(__file__).resolve().parent
_TRAINING = _LABELING.parent
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))  # `import labeling`
sys.path.insert(0, str(_REPO / "active_pipeline"))  # `import smoothie_cv`
from labeling import db

app = Flask(__name__, template_folder=str(db.ROOT / "templates"),
            static_folder=str(db.ROOT / "static"))


def _check_mode(mode: str | None) -> str:
    if mode not in db.MODES:
        abort(400, f"mode must be one of {db.MODES}")
    return mode


@app.route("/")
def index() -> str:
    return send_from_directory(str(db.ROOT / "templates"), "label_review.html")


@app.route("/image/<int:file_id>")
def image(file_id: int):
    path = db.IMAGES_DIR / f"{file_id}.jpg"
    if not path.exists():
        abort(404)
    return send_file(str(path), mimetype="image/jpeg")


def _predictions_for(conn, file_id: int, mode: str):
    """(polygons, confidences) as parallel lists, ordered by descending conf."""
    rows = conn.execute(
        "SELECT polygon, confidence FROM predictions "
        "WHERE file_id = ? AND mode = ? ORDER BY confidence DESC",
        (file_id, mode),
    ).fetchall()
    polys = [json.loads(r["polygon"]) for r in rows]
    confs = [r["confidence"] for r in rows]
    return polys, confs


def _annotations_for(conn, file_id: int, mode: str):
    rows = conn.execute(
        "SELECT polygon FROM annotations WHERE file_id = ? AND mode = ? ORDER BY id",
        (file_id, mode),
    ).fetchall()
    return [json.loads(r["polygon"]) for r in rows]


def _item_payload(conn, file_id: int, mode: str) -> dict:
    """Editable shapes for one image: the saved annotations once decided,
    otherwise the model's predictions (with per-shape confidence)."""
    rs = conn.execute(
        "SELECT status FROM review_status WHERE file_id = ? AND mode = ?",
        (file_id, mode),
    ).fetchone()
    ms = conn.execute(
        "SELECT status FROM mode_status WHERE file_id = ? AND mode = ?",
        (file_id, mode),
    ).fetchone()
    review = rs["status"] if rs else None
    # Show saved annotations for an already-decided image; else the prediction.
    if ms is not None:
        shapes = _annotations_for(conn, file_id, mode)
        confs = [None] * len(shapes)
        predicted = False
    else:
        shapes, confs = _predictions_for(conn, file_id, mode)
        predicted = True
    return {
        "file_id": file_id,
        "mode": mode,
        "review_status": review,
        "mode_status": ms["status"] if ms else None,
        "shapes": shapes,
        "confidences": confs,
        "predicted": predicted,
    }


def _pending_ordered(conn, mode: str, sort: str) -> list[tuple[int, float]]:
    """Pending (file_id, representative_conf) in review order.

    representative_conf = max prediction confidence for the image, or -1.0 when
    the model detected nothing (those sort FIRST under conf_asc so the reviewer
    checks likely false-negatives). ``sort='file'`` orders by file_id only.
    """
    rows = conn.execute(
        "SELECT rs.file_id, COALESCE(MAX(p.confidence), -1.0) AS conf "
        "FROM review_status rs "
        "LEFT JOIN predictions p "
        "  ON p.file_id = rs.file_id AND p.mode = rs.mode "
        "LEFT JOIN image_flags fl ON fl.file_id = rs.file_id "
        "WHERE rs.mode = ? AND rs.status = 'pending' "
        "  AND fl.file_id IS NULL "  # exclude no_smoothie / machinery shots
        "GROUP BY rs.file_id",
        (mode,),
    ).fetchall()
    items = [(r["file_id"], r["conf"]) for r in rows]
    if sort == "file":
        items.sort(key=lambda t: t[0])
    else:  # conf_asc (default)
        items.sort(key=lambda t: (t[1], t[0]))
    return items


def _conf_for(conn, file_id: int, mode: str) -> float:
    row = conn.execute(
        "SELECT COALESCE(MAX(confidence), -1.0) c FROM predictions "
        "WHERE file_id = ? AND mode = ?",
        (file_id, mode),
    ).fetchone()
    return row["c"] if row else -1.0


@app.route("/api/review/next")
def api_next():
    """Next PENDING image in review order. ``?after=<file_id>`` continues past a
    just-decided image (which is no longer pending) using its stored confidence."""
    mode = _check_mode(request.args.get("mode"))
    sort = request.args.get("sort", "conf_asc")
    after = request.args.get("after", type=int)
    conn = db.connect()
    ordered = _pending_ordered(conn, mode, sort)
    ordered = [(f, c) for (f, c) in ordered
               if (db.IMAGES_DIR / f"{f}.jpg").exists()]
    if not ordered:
        return jsonify({"done": True})
    if after is None:
        fid = ordered[0][0]
        return jsonify(_item_payload(conn, fid, mode))
    # Find the first entry strictly after `after` in the active ordering.
    a_conf = _conf_for(conn, after, mode)
    if sort == "file":
        nxt = next((f for (f, _) in ordered if f > after), None)
    else:
        nxt = next((f for (f, c) in ordered if (c, f) > (a_conf, after)), None)
    if nxt is None:
        return jsonify({"done": True})
    return jsonify(_item_payload(conn, nxt, mode))


@app.route("/api/review/item/<int:file_id>")
def api_item(file_id: int):
    mode = _check_mode(request.args.get("mode"))
    conn = db.connect()
    if conn.execute("SELECT 1 FROM files WHERE file_id = ?",
                    (file_id,)).fetchone() is None:
        abort(404)
    return jsonify(_item_payload(conn, file_id, mode))


@app.route("/api/review/seek")
def api_seek():
    """Adjacent image that has a review_status row (any status), by file_id, so
    you can go back and re-decide an image reviewed earlier. ``dir`` in prev|next."""
    mode = _check_mode(request.args.get("mode"))
    file_id = request.args.get("file_id", type=int)
    direction = request.args.get("dir", "prev")
    if file_id is None or direction not in ("prev", "next"):
        abort(400, "file_id required and dir must be 'prev' or 'next'")
    op, order = ("<", "DESC") if direction == "prev" else (">", "ASC")
    # Traverse ALL predicted images in this mode (any status) in file order, so
    # you can step back to a decided image and change it. Machinery shots excluded.
    sql = (
        f"SELECT rs.file_id FROM review_status rs "
        f"LEFT JOIN image_flags fl ON fl.file_id = rs.file_id "
        f"WHERE rs.mode = ? AND fl.file_id IS NULL AND rs.file_id {op} ? "
        f"ORDER BY rs.file_id {order} LIMIT 1"
    )
    conn = db.connect()
    cur = file_id
    while True:
        row = conn.execute(sql, (mode, cur)).fetchone()
        if row is None:
            return jsonify({"edge": True})
        if (db.IMAGES_DIR / f"{row['file_id']}.jpg").exists():
            return jsonify(_item_payload(conn, row["file_id"], mode))
        cur = row["file_id"]


def _push_priority(mode: str, file_id: int) -> None:
    """Append a rejected id to labeling/priority/<mode>.txt so the hand pipeline
    (app_multi.py) serves it first. No-op if already present."""
    path = db.ROOT / "priority" / f"{mode}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[int] = set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                try:
                    existing.add(int(line))
                except ValueError:
                    pass
    if file_id not in existing:
        with path.open("a") as fh:
            fh.write(f"{file_id}  # rejected in review\n")


def _pop_priority(mode: str, file_id: int) -> None:
    """Remove a file_id from labeling/priority/<mode>.txt — used when a reject is
    reversed to accept/clean, so it no longer jumps the hand-labeler queue."""
    path = db.ROOT / "priority" / f"{mode}.txt"
    if not path.exists():
        return
    kept = []
    for line in path.read_text().splitlines():
        body = line.split("#", 1)[0].strip()
        try:
            if body and int(body) == file_id:
                continue  # drop this line
        except ValueError:
            pass
        kept.append(line)
    path.write_text("\n".join(kept) + ("\n" if kept else ""))


@app.route("/api/review/decide", methods=["POST"])
def api_decide():
    """Persist a review decision.

    Body: {file_id, mode, decision in (approve|clean|reject), shapes:[{polygon}]}
      approve -> annotations(source='model') + mode_status='labeled'  (needs >=1 poly)
      clean   -> mode_status='clean' (empty negative label), no shapes
      reject  -> review_status='rejected' only; no mode_status row (hand re-serves)
    """
    data = request.get_json(force=True)
    file_id = data.get("file_id")
    mode = _check_mode(data.get("mode"))
    decision = data.get("decision")
    if file_id is None or decision not in ("approve", "clean", "reject"):
        abort(400, "file_id required and decision in approve|clean|reject")

    conn = db.connect()

    if decision == "approve":
        polys: list[list[list[int]]] = []
        for s in data.get("shapes") or []:
            pts = s.get("polygon") if isinstance(s, dict) else s
            if pts and len(pts) >= 3:
                polys.append([[int(x), int(y)] for x, y in pts])
        if not polys:
            abort(400, "approve requires at least one polygon with >=3 points")
        conn.execute("DELETE FROM annotations WHERE file_id = ? AND mode = ?",
                     (file_id, mode))
        for pts in polys:
            conn.execute(
                "INSERT INTO annotations (file_id, mode, polygon, source, "
                "created_at) VALUES (?, ?, ?, ?, datetime('now'))",
                (file_id, mode, json.dumps(pts), db.SOURCE_MODEL),
            )
        _set_mode_status(conn, file_id, mode, "labeled")
        _set_review_status(conn, file_id, mode, "approved")
        _pop_priority(mode, file_id)  # in case this reverses an earlier reject

    elif decision == "clean":
        conn.execute("DELETE FROM annotations WHERE file_id = ? AND mode = ?",
                     (file_id, mode))
        _set_mode_status(conn, file_id, mode, "clean")
        _set_review_status(conn, file_id, mode, "approved")
        _pop_priority(mode, file_id)  # in case this reverses an earlier reject

    else:  # reject -> leave undecided for the hand pipeline
        conn.execute("DELETE FROM annotations WHERE file_id = ? AND mode = ?",
                     (file_id, mode))
        conn.execute("DELETE FROM mode_status WHERE file_id = ? AND mode = ?",
                     (file_id, mode))
        _set_review_status(conn, file_id, mode, "rejected")
        _push_priority(mode, file_id)

    conn.commit()
    return jsonify({"ok": True})


def _set_mode_status(conn, file_id: int, mode: str, status: str) -> None:
    conn.execute(
        "INSERT INTO mode_status (file_id, mode, status, updated_at) "
        "VALUES (?, ?, ?, datetime('now')) "
        "ON CONFLICT(file_id, mode) DO UPDATE SET "
        "status=excluded.status, updated_at=excluded.updated_at",
        (file_id, mode, status),
    )


def _set_review_status(conn, file_id: int, mode: str, status: str) -> None:
    conn.execute(
        "INSERT INTO review_status (file_id, mode, status, reviewed_at) "
        "VALUES (?, ?, ?, datetime('now')) "
        "ON CONFLICT(file_id, mode) DO UPDATE SET "
        "status=excluded.status, reviewed_at=excluded.reviewed_at",
        (file_id, mode, status),
    )


@app.route("/api/review/progress")
def api_progress():
    """Review counts for ``mode``: pending / approved / rejected."""
    mode = _check_mode(request.args.get("mode"))
    conn = db.connect()
    counts = {s: 0 for s in db.REVIEW_STATUSES}
    # Exclude no_smoothie / machinery shots so pending reflects what's actually served.
    for r in conn.execute(
        "SELECT rs.status, COUNT(*) c FROM review_status rs "
        "LEFT JOIN image_flags fl ON fl.file_id = rs.file_id "
        "WHERE rs.mode = ? AND fl.file_id IS NULL GROUP BY rs.status",
        (mode,),
    ):
        if r["status"] in counts:
            counts[r["status"]] = r["c"]
    counts["total"] = sum(counts[s] for s in db.REVIEW_STATUSES)
    excluded = conn.execute(
        "SELECT COUNT(*) c FROM review_status rs "
        "JOIN image_flags fl ON fl.file_id = rs.file_id WHERE rs.mode = ?",
        (mode,),
    ).fetchone()["c"]
    counts["no_smoothie"] = excluded
    return jsonify({"mode": mode, "counts": counts})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the model-assisted review UI")
    parser.add_argument("--mode", default="spill", choices=db.MODES,
                        help="default mode the UI opens in (also selectable in URL)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    db.ensure_dirs()
    db.connect().close()
    app.config["DEFAULT_MODE"] = args.mode
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
