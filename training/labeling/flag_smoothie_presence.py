"""Flag machinery / empty-rig shots (no smoothie in frame) so they are excluded
from every pipeline — review queues, the hand labeler, and export.

Uses the trained CONTAINER/standard detector (checkpoints/yolo_standard_seg.pt)
as a "is there a smoothie here at all?" gate: an image with ZERO smoothie
detections at/above --conf is flagged 'no_smoothie' in the image_flags table.
Images where a smoothie IS found get any stale flag cleared (idempotent /
self-correcting, so re-running after a better container model un-hides recovered
cups).

Run under the conda python (CPU; MPS segfaults on seg):
  /opt/miniconda3/bin/python labeling/flag_smoothie_presence.py
  /opt/miniconda3/bin/python labeling/flag_smoothie_presence.py --conf 0.30 --limit 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

_LABELING = Path(__file__).resolve().parent
_TRAINING = _LABELING.parent
_REPO = _TRAINING.parent
sys.path.insert(0, str(_TRAINING))  # `import labeling`
sys.path.insert(0, str(_REPO / "active_pipeline"))  # `import smoothie_cv`
from labeling import db

# The container detector historically deploys here (see CLAUDE.md / config).
CONTAINER_WEIGHTS = db.CHECKPOINTS_DIR / "yolo_standard_seg.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(CONTAINER_WEIGHTS),
                    help="container/smoothie detector weights (presence gate)")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="min confidence to count as a smoothie detection")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-flagged", action="store_true",
                    help="skip images already flagged no_smoothie (faster re-run)")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"container weights not found: {weights}")

    from ultralytics import YOLO

    model = YOLO(str(weights))
    conn = db.connect()

    rows = conn.execute(
        "SELECT file_id FROM files WHERE downloaded = 1 ORDER BY file_id"
    ).fetchall()
    ids = [r["file_id"] for r in rows]
    if args.skip_flagged:
        flagged = {r["file_id"] for r in conn.execute(
            "SELECT file_id FROM image_flags WHERE flag = ?", (db.NO_SMOOTHIE,))}
        ids = [i for i in ids if i not in flagged]
    if args.limit:
        ids = ids[: args.limit]
    print(f"scanning {len(ids)} images for smoothie presence (conf≥{args.conf})")

    n_no = n_yes = 0
    for i, fid in enumerate(ids, 1):
        p = db.IMAGES_DIR / f"{fid}.jpg"
        if not p.exists():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        result = model(img, verbose=False, device="cpu", conf=args.conf)[0]
        confs = (result.boxes.conf.cpu().numpy().tolist()
                 if result.boxes is not None and len(result.boxes) else [])
        top = max(confs, default=0.0)
        has_smoothie = len(confs) > 0

        if has_smoothie:
            conn.execute("DELETE FROM image_flags WHERE file_id = ?", (fid,))
            n_yes += 1
        else:
            conn.execute(
                "INSERT INTO image_flags (file_id, flag, confidence, created_at) "
                "VALUES (?, ?, ?, datetime('now')) "
                "ON CONFLICT(file_id) DO UPDATE SET "
                "flag=excluded.flag, confidence=excluded.confidence, "
                "created_at=excluded.created_at",
                (fid, db.NO_SMOOTHIE, top),
            )
            n_no += 1
        conn.commit()
        if i % 50 == 0 or not has_smoothie:
            print(f"[{i}/{len(ids)}] {fid}  smoothie={'yes' if has_smoothie else 'NO'}  top={top:.2f}")

    print(f"done: {n_yes} have a smoothie, {n_no} flagged no_smoothie (excluded)")


if __name__ == "__main__":
    main()
