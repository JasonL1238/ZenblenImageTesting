"""Build a YOLO-seg dataset of the INVERSE of the chunk labels.

Target class 'blended' = the liquid ROI MINUS the human-labeled chunks, i.e.
"the interior of the smoothie that is NOT an unblended lump" — exactly the
reverse-label idea:
    ROI (yolo_standard_seg, model 1)  -  chunks (human labels, annotations)  =  blended

For chunk-'clean' images (no chunks) the target is the whole ROI. For
chunk-'labeled' images the chunks become HOLES in the blended region. YOLO-seg
polygons cannot store holes, so each hole is bridged into the outer contour with
a zero-width slit -> a single simple polygon that still excludes the chunk area.

NOTE (documented, not hidden): the training signal here is 100% derived from the
ROI + existing chunk labels, so a model can at best reproduce (ROI - chunks). It
inherits the chunk blind spot: streaks with no chunk label are inside 'blended',
teaching the model streaks are blended. This is the experiment the user asked
for; the numbers will show what it can and cannot learn.

Run (needs YOLO/torch -> conda python):
  /opt/miniconda3/bin/python build_inverse_dataset.py --selftest   # verify bridging
  /opt/miniconda3/bin/python build_inverse_dataset.py              # build dataset
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "texture_blendedness"))
import common  # noqa: E402  (get_masks + cached ROI)

REPO = common.REPO
DB = REPO / "training" / "labeling" / "labels.db"
OUT = REPO / "training" / "labeling" / "datasets" / "blended_dataset"
APPROX_EPS = 1.5      # polygon simplification (px) — keeps point counts sane
MIN_AREA = 200        # drop tiny blended fragments (px)


def chunk_decided():
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT file_id, status FROM mode_status "
        "WHERE mode='chunk' AND status IN ('labeled','clean') ORDER BY file_id"
    ).fetchall()
    out = []
    for r in rows:
        if (common.IMAGES_DIR / f"{r['file_id']}.jpg").exists():
            out.append((r["file_id"], r["status"]))
    return out


def chunk_polys(conn, file_id):
    return [json.loads(r["polygon"]) for r in conn.execute(
        "SELECT polygon FROM annotations WHERE file_id=? AND mode='chunk' ORDER BY id",
        (file_id,))]


def _bridge(outer, holes):
    """Merge hole loops into an outer loop via nearest-point zero-width slits,
    yielding one simple (self-touching) polygon that excludes each hole."""
    poly = [tuple(map(int, p)) for p in outer]
    for hole in holes:
        h = [tuple(map(int, p)) for p in hole]
        if len(h) < 3:
            continue
        # nearest (i in poly, j in hole)
        pa = np.array(poly)
        ha = np.array(h)
        d2 = ((pa[:, None, :] - ha[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(d2.argmin()), d2.shape)
        loop = h[j:] + h[:j] + [h[j]]              # hole traversed back to start
        poly = poly[:i + 1] + loop + [poly[i]] + poly[i + 1:]
    return poly


def mask_to_polys(mask):
    """Binary mask -> list of YOLO polygons (each a flat [x,y,...] pixel list),
    holes bridged into their parent outer contour."""
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    hier = hier[0]
    polys = []
    for idx, c in enumerate(cnts):
        if hier[idx][3] != -1:          # a hole; handled with its parent
            continue
        if cv2.contourArea(c) < MIN_AREA:
            continue
        outer = cv2.approxPolyDP(c, APPROX_EPS, True).reshape(-1, 2)
        if len(outer) < 3:
            continue
        holes = []
        child = hier[idx][2]
        while child != -1:
            hc = cnts[child]
            if cv2.contourArea(hc) >= MIN_AREA:
                hp = cv2.approxPolyDP(hc, APPROX_EPS, True).reshape(-1, 2)
                if len(hp) >= 3:
                    holes.append(hp)
            child = hier[child][0]
        merged = _bridge(outer, holes) if holes else [tuple(map(int, p)) for p in outer]
        polys.append([v for xy in merged for v in xy])
    return polys


def _rasterize(polys, shape):
    m = np.zeros(shape, np.uint8)
    for p in polys:
        pts = np.array(p, np.int32).reshape(-1, 2)
        cv2.fillPoly(m, [pts], 255)
    return m


def selftest():
    """Donut: outer 100x100 square with a 30x30 hole. Bridged polygon must
    reproduce the holed region (high IoU), NOT fill the hole back in."""
    m = np.zeros((160, 160), np.uint8)
    cv2.rectangle(m, (30, 30), (130, 130), 255, -1)
    cv2.rectangle(m, (65, 65), (95, 95), 0, -1)   # the hole (a "chunk")
    polys = mask_to_polys(m > 0)
    rr = _rasterize(polys, m.shape)
    inter = ((rr > 0) & (m > 0)).sum()
    union = ((rr > 0) | (m > 0)).sum()
    hole_filled = ((rr > 0) & (m == 0)).sum()      # px wrongly filled in the hole
    print(f"selftest IoU={inter/union:.3f}  hole-fill-leak={hole_filled}px "
          f"(want IoU~1.0, leak~0)")
    return inter / union > 0.95 and hole_filled < 120


def _yolo_lines(polys, w, h):
    out = []
    for p in polys:
        coords = " ".join(f"{p[k]/w:.6f} {p[k+1]/h:.6f}" for k in range(0, len(p), 2))
        out.append("0 " + coords)
    return out


def _split(ids, val, test):
    n = len(ids)
    nt = max(0, round(n * test))
    nv = max(0, round(n * val))
    return ids[:n-nv-nt], ids[n-nv-nt:n-nt], ids[n-nt:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        ok = selftest()
        print("SELFTEST", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)

    items = chunk_decided()
    if args.limit:
        items = items[:args.limit]
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    for sub in ("images", "labels"):
        if (OUT / sub).exists():
            shutil.rmtree(OUT / sub)
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    ids = [fid for fid, _ in items]
    tr, va, te = _split(ids, args.val, args.test)
    split_of = {**{i: "train" for i in tr}, **{i: "val" for i in va},
                **{i: "test" for i in te}}

    counts = {"train": 0, "val": 0, "test": 0}
    empty = 0
    for k, (fid, status) in enumerate(items):
        path = common.IMAGES_DIR / f"{fid}.jpg"
        image, roi, _logo = common.get_masks(path)
        h, w = roi.shape
        blended = (roi > 0)
        if status == "labeled":
            chunks = _rasterize(
                [[v for xy in poly for v in xy] for poly in chunk_polys(conn, fid)],
                (h, w)) > 0
            blended = blended & ~chunks
        polys = mask_to_polys(blended)
        split = split_of[fid]
        shutil.copyfile(path, OUT / "images" / split / f"blended_{fid}.jpg")
        lines = _yolo_lines(polys, w, h)
        (OUT / "labels" / split / f"blended_{fid}.txt").write_text(
            ("\n".join(lines) + "\n") if lines else "")
        if not lines:
            empty += 1
        counts[split] += 1
        if (k + 1) % 50 == 0:
            print(f"  ...{k+1}/{len(items)}", flush=True)

    (OUT / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\n"
        "nc: 1\nnames:\n  0: blended\n")
    print(f"built {sum(counts.values())} -> {OUT}")
    print(f"  train={counts['train']} val={counts['val']} test={counts['test']} "
          f"({empty} empty)")


if __name__ == "__main__":
    main()
