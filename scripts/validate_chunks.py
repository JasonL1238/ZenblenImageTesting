"""Run the chunk detector on every image and write a clean, human-readable report.

ROIs come from the cached YOLO-seg masks by default (fast + deterministic —
build with scripts/cache_yolo_rois.py). Pass --live to run the model instead,
or --roi-cache to point at another YOLO ROI cache.

Verdicts are diffed against the previous report's scores.csv (--baseline) so
every run reports exactly what flipped — new flags, lost flags.

Outputs (under --out, default outputs/report/):
  flagged.png    — flagged smoothies original vs detection, worst first
  overlays/      — per-image triptych (original | ROI | chunk detection)
  scores.csv     — per-image blend score + verdict
  README.md      — summary + diff vs baseline

Usage:
  /opt/miniconda3/bin/python scripts/validate_chunks.py
  /opt/miniconda3/bin/python scripts/validate_chunks.py --live --weights runs/smoothie-seg/nano-v4/weights/best.pt --out outputs/report_v4
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoothie_cv.config import Config
from smoothie_cv.pipelines.classical_cv import ClassicalCVPipeline
from smoothie_cv.scoring.metrics import overlay_mask

IMG_DIRS      = [Path("data/images/red_pink"), Path("data/images/yellow")]
FLAG_SCORE    = 0.999   # a smoothie is "flagged" below this blend score
PAIR_W        = 230     # width of each image in the side-by-side pair
PAIRS_PER_ROW = 3


def load_baseline(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path) as f:
        return {r["stem"]: r["verdict"] for r in csv.DictReader(f)}


def _label(img, lines, color):
    cv2.rectangle(img, (0, 0), (img.shape[1], 16 + 18 * len(lines)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (4, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img


def _roi_strip(img, roi_mask):
    """Visualize the ROI: dim everything outside the mask, boundary in green."""
    vis = img.astype(np.float32)
    vis[roi_mask == 0] *= 0.30
    vis = vis.astype(np.uint8)
    cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    return vis


def _triptych(img, roi_mask, chunk_mask, score, px):
    """[ original | ROI mask | chunk detection ] for one smoothie."""
    orig    = _label(img.copy(), ["original"], (255, 255, 255))
    roi_vis = _label(_roi_strip(img, roi_mask), ["ROI mask"], (0, 255, 0))
    det     = overlay_mask(img, chunk_mask, color=(255, 0, 0), alpha=0.55)
    verdict = "CHUNKS" if score < FLAG_SCORE else "clean"
    det     = _label(det, [f"detect: {verdict}", f"blend {score:.3f}  {px}px"], (80, 160, 255))
    gap     = np.full((img.shape[0], 6, 3), 40, np.uint8)
    return np.hstack([orig, gap, roi_vis, gap, det])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roi-cache", default="outputs/roi_cache_yolo",
                    help="dir of cached <stem>.png ROI masks (scripts/cache_yolo_rois.py)")
    ap.add_argument("--live", action="store_true",
                    help="run the YOLO model instead of using the ROI cache")
    ap.add_argument("--weights", default="checkpoints/yolo_smoothie_seg.pt",
                    help="YOLO weights for --live")
    ap.add_argument("--out", default="outputs/report",
                    help="report output dir")
    ap.add_argument("--baseline", default="outputs/report/scores.csv",
                    help="previous scores.csv to diff verdicts against")
    ap.add_argument("--logo-yolo", action="store_true",
                    help="enable trained-logo-mask chunk suppression "
                         "(dev_logo_yolo_suppress); runs the logo model live per image")
    ap.add_argument("--logo-weights", default=None,
                    help="logo YOLO weights (default config.logo_weights)")
    args = ap.parse_args()

    out      = Path(args.out)
    overlays = out / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)

    # load the baseline BEFORE writing anything (out may equal the baseline dir)
    baseline = load_baseline(Path(args.baseline))

    model = None
    roi_cache = Path(args.roi_cache)
    if args.live:
        from ultralytics import YOLO
        weights = Path(args.weights)
        if not weights.exists():
            print(f"ERROR: weights not found: {weights}")
            sys.exit(1)
        print(f"Loading YOLO ({weights}) …")
        model = YOLO(str(weights))
    elif not roi_cache.is_dir():
        print(f"ERROR: ROI cache not found: {roi_cache} — "
              f"build it with scripts/cache_yolo_rois.py or pass --live")
        sys.exit(1)

    cfg  = Config()
    if args.logo_yolo:
        cfg.dev_logo_yolo_suppress = True
        if args.logo_weights:
            cfg.logo_weights = Path(args.logo_weights)
        if not Path(cfg.logo_weights).exists():
            print(f"ERROR: logo weights not found: {cfg.logo_weights}")
            sys.exit(1)
        print(f"Logo suppression ON (logo weights: {cfg.logo_weights})")
    pipe = ClassicalCVPipeline(cfg)
    imgs = sorted(p for d in IMG_DIRS for p in d.glob("*.jpg"))

    rows      = []
    originals = {}

    for i, p in enumerate(imgs, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]

        if model is not None:
            from smoothie_cv.detection.yolo import get_yolo_roi
            roi = get_yolo_roi(model(img, verbose=False)[0], (h, w))
        else:
            cached = roi_cache / f"{p.stem}.png"
            roi = (cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE)
                   if cached.exists() else None)
            if roi is None:
                roi = np.zeros((h, w), dtype=np.uint8)

        if roi.sum() == 0:
            print(f"  [{i:2d}] WARNING no ROI: {p.stem[:40]}")
            rows.append({"stem": p.stem, "shade": p.parent.name,
                         "score": 1.0, "chunk_pixels": 0, "verdict": "clean",
                         "baseline_verdict": baseline.get(p.stem, "unknown"),
                         "flipped": False})
            continue

        r  = pipe.analyze(img, roi)
        px = int((r.mask > 0).sum())
        verdict = "chunks" if r.blend_score < FLAG_SCORE else "clean"

        base_v = baseline.get(p.stem, "unknown")
        flip   = "" if base_v in (verdict, "unknown") else f"  ← WAS {base_v.upper()}"

        rows.append({"stem": p.stem, "shade": p.parent.name,
                     "score": round(r.blend_score, 4),
                     "chunk_pixels": px, "verdict": verdict,
                     "baseline_verdict": base_v,
                     "flipped": base_v not in (verdict, "unknown")})
        print(f"  [{i:2d}/{len(imgs)}] {verdict:6s}  score={r.blend_score:.3f}  {p.stem[:36]}{flip}")

        cv2.imwrite(str(overlays / f"{p.stem}.png"),
                    _triptych(img, roi, r.mask, r.blend_score, px))
        originals[p.stem] = (img, overlay_mask(img, r.mask, color=(255, 0, 0), alpha=0.55),
                             r.blend_score, px)

    # ── flagged.png: only the flagged smoothies, sorted worst-first ──
    flagged = sorted([r for r in rows if r["verdict"] == "chunks"],
                     key=lambda r: r["score"])
    cells = []
    for rank, r in enumerate(flagged, 1):
        if r["stem"] not in originals:
            continue
        img, vis, score, px = originals[r["stem"]]

        def fit(im):
            h2, w2 = im.shape[:2]
            return cv2.resize(im, (PAIR_W, int(h2 * PAIR_W / w2)))

        orig_f, over_f = fit(img), fit(vis)
        short = r["stem"].replace("UserGrab_", "")[:8]
        note  = " [NEW]" if r["baseline_verdict"] == "clean" else ""
        orig_f = _label(orig_f, [f"#{rank}  {short}{note}", "original"], (255, 255, 255))
        over_f = _label(over_f, [f"blend {score:.3f}", f"{px}px detected"], (80, 160, 255))
        cells.append(np.hstack([orig_f, over_f]))

    if cells:
        ch    = max(c.shape[0] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                 for c in cells]
        gap   = 8
        grid_rows = []
        for i in range(0, len(cells), PAIRS_PER_ROW):
            row = cells[i:i + PAIRS_PER_ROW]
            while len(row) < PAIRS_PER_ROW:
                row.append(np.zeros_like(cells[0]))
            row_img = row[0]
            for c in row[1:]:
                sep = np.full((row_img.shape[0], gap, 3), 40, np.uint8)
                row_img = np.hstack([row_img, sep, c])
            grid_rows.append(row_img)
        montage = grid_rows[0]
        for gr in grid_rows[1:]:
            sep = np.full((gap, montage.shape[1], 3), 40, np.uint8)
            montage = np.vstack([montage, sep, gr])
        cv2.imwrite(str(out / "flagged.png"), montage)

    # ── scores.csv ──
    with open(out / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "shade", "score",
                                               "chunk_pixels", "verdict",
                                               "baseline_verdict", "flipped"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["score"]))

    # ── README.md: summary + diff vs baseline ──
    n_flag    = len(flagged)
    n_total   = len(rows)
    flipped   = [r for r in rows if r["flipped"]]
    new_fp    = [r for r in flipped if r["verdict"] == "chunks"]   # clean→chunks
    lost_tp   = [r for r in flipped if r["verdict"] == "clean"]    # chunks→clean
    base_flag = sum(1 for v in baseline.values() if v == "chunks")

    roi_src = f"live {args.weights}" if args.live else args.roi_cache
    summary = f"""# Chunk-detection report — ROIs: {roi_src}
## {n_total} images · {n_flag} flagged · {n_total - n_flag} clean
## baseline: {base_flag} flagged  →  this run: {n_flag} flagged  (Δ {n_flag - base_flag:+d})

### Verdict flips vs baseline ({len(flipped)} total)
- New flags (clean → chunks): {len(new_fp)}
- Lost flags (chunks → clean): {len(lost_tp)}

#### New flags — were clean under the baseline
"""
    for r in sorted(new_fp, key=lambda x: x["score"]):
        summary += f"- {r['stem'][:50]}  score={r['score']}  shade={r['shade']}\n"

    summary += "\n#### Lost flags — were chunky under the baseline\n"
    for r in sorted(lost_tp, key=lambda x: x["score"], reverse=True):
        summary += f"- {r['stem'][:50]}  score={r['score']}  shade={r['shade']}\n"

    summary += """
## All flagged smoothies (worst first)
| rank | image | score | px | was |
|---|---|---|---|---|
"""
    for rank, r in enumerate(flagged, 1):
        note = " **NEW**" if r["baseline_verdict"] == "clean" else ""
        summary += (f"| {rank} | {r['stem'][:28]} | {r['score']:.3f} "
                    f"| {r['chunk_pixels']} | {r['baseline_verdict']}{note} |\n")

    (out / "README.md").write_text(summary)
    print(f"\n{'='*60}")
    print(f"baseline:  {base_flag} flagged")
    print(f"this run:  {n_flag} flagged  (Δ {n_flag - base_flag:+d})")
    print(f"  New flags:  {len(new_fp)}")
    print(f"  Lost flags: {len(lost_tp)}")
    print(f"{'='*60}")
    print(f"open → {out / 'flagged.png'}")
    print(f"       {out / 'README.md'}")


if __name__ == "__main__":
    main()
