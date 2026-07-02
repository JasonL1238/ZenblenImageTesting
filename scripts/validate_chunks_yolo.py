"""Run the chunk detector using YOLO-seg ROIs (instead of cached SAM masks).

Compares verdicts against the SAM baseline (outputs/report/scores.csv) and
reports what flipped — new FPs, recovered TPs, or regressions.

Outputs (under outputs/report_yolo_<tag>/):
  flagged.png    — flagged smoothies original vs detection, worst first
  overlays/      — per-image triptych (original | ROI | chunk detection)
  scores.csv     — per-image blend score + verdict
  README.md      — summary + diff vs SAM baseline

Usage:
  /opt/miniconda3/bin/python scripts/validate_chunks_yolo.py
  /opt/miniconda3/bin/python scripts/validate_chunks_yolo.py --weights runs/smoothie-seg/nano-v2/weights/best.pt --tag v2
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoothie_cv.config import Config
from smoothie_cv.pipelines.classical_cv import ClassicalCVPipeline
from smoothie_cv.scoring.metrics import overlay_mask

SAM_BASELINE  = Path("outputs/report/scores.csv")
IMG_DIRS      = [Path("data/images/red_pink"), Path("data/images/yellow")]
FLAG_SCORE    = 0.999
PAIR_W        = 230
PAIRS_PER_ROW = 3


def get_yolo_roi(result, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if result.masks is None or len(result.masks) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    confs = result.boxes.conf.cpu().numpy()
    idx   = int(np.argmax(confs))
    raw   = result.masks.data[idx].cpu().numpy()
    m     = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((m > 0.5) * 255).astype(np.uint8)


def load_sam_baseline(path: Path) -> dict[str, str]:
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
    vis = img.astype(np.float32)
    vis[roi_mask == 0] *= 0.30
    vis = vis.astype(np.uint8)
    cnts, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 0), 2)
    return vis


def _triptych(img, roi_mask, chunk_mask, score, px):
    orig    = _label(img.copy(), ["original"], (255, 255, 255))
    roi_vis = _label(_roi_strip(img, roi_mask), ["YOLO ROI"], (0, 255, 0))
    det     = overlay_mask(img, chunk_mask, color=(255, 0, 0), alpha=0.55)
    verdict = "CHUNKS" if score < FLAG_SCORE else "clean"
    det     = _label(det, [f"detect: {verdict}", f"blend {score:.3f}  {px}px"], (80, 160, 255))
    gap     = np.full((img.shape[0], 6, 3), 40, np.uint8)
    return np.hstack([orig, gap, roi_vis, gap, det])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/smoothie-seg/nano-v3/weights/best.pt")
    ap.add_argument("--tag",     default="v3",
                    help="label for output folder (outputs/report_yolo_<tag>/)")
    args = ap.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: weights not found: {weights}")
        sys.exit(1)

    out      = Path(f"outputs/report_yolo_{args.tag}")
    overlays = out / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO {args.tag} ({weights}) …")
    model = YOLO(str(weights))
    cfg   = Config()
    pipe  = ClassicalCVPipeline(cfg)

    sam_baseline = load_sam_baseline(SAM_BASELINE)
    imgs = sorted(p for d in IMG_DIRS for p in d.glob("*.jpg"))

    rows      = []
    originals = {}

    for i, p in enumerate(imgs, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]

        yolo_roi = get_yolo_roi(model(img, verbose=False)[0], (h, w))
        no_det   = yolo_roi.sum() == 0

        if no_det:
            # fall back: mark as clean with a warning
            print(f"  [{i:2d}] WARNING no YOLO detection: {p.stem[:40]}")
            rows.append({"stem": p.stem, "shade": p.parent.name,
                         "score": 1.0, "chunk_pixels": 0,
                         "verdict": "clean", "roi_source": "none"})
            continue

        r  = pipe.analyze(img, yolo_roi)
        px = int((r.mask > 0).sum())
        verdict = "chunks" if r.blend_score < FLAG_SCORE else "clean"

        sam_v = sam_baseline.get(p.stem, "unknown")
        flip  = "" if sam_v == verdict else f"  ← WAS {sam_v.upper()}"

        rows.append({"stem": p.stem, "shade": p.parent.name,
                     "score": round(r.blend_score, 4),
                     "chunk_pixels": px, "verdict": verdict,
                     "sam_verdict": sam_v, "flipped": sam_v != verdict})
        print(f"  [{i:2d}/{len(imgs)}] {verdict:6s}  score={r.blend_score:.3f}  {p.stem[:36]}{flip}")

        cv2.imwrite(str(overlays / f"{p.stem}.png"),
                    _triptych(img, yolo_roi, r.mask, r.blend_score, px))
        originals[p.stem] = (img, overlay_mask(img, r.mask, color=(255, 0, 0), alpha=0.55),
                             r.blend_score, px)

    # flagged.png montage
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
        sam_v = r.get("sam_verdict", "?")
        note  = " [NEW]" if sam_v == "clean" else ""
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

    # CSV
    with open(out / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["stem", "shade", "score",
                                               "chunk_pixels", "verdict",
                                               "sam_verdict", "flipped"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["score"]))

    # diff vs SAM
    n_flag   = len(flagged)
    n_total  = len(rows)
    flipped  = [r for r in rows if r.get("flipped")]
    new_fp   = [r for r in flipped if r["verdict"] == "chunks"]   # clean→chunks
    lost_tp  = [r for r in flipped if r["verdict"] == "clean"]    # chunks→clean
    sam_flag = sum(1 for v in sam_baseline.values() if v == "chunks")

    summary = f"""# Chunk-detection report — YOLO {args.tag} ROIs
## {n_total} images · {n_flag} flagged · {n_total - n_flag} clean
## SAM baseline: {sam_flag} flagged  →  YOLO {args.tag}: {n_flag} flagged  (Δ {n_flag - sam_flag:+d})

### Verdict flips vs SAM baseline ({len(flipped)} total)
- New flags (clean → chunks): {len(new_fp)}
- Lost flags (chunks → clean): {len(lost_tp)}

#### New flags — were clean under SAM, now chunky under YOLO ROI
"""
    for r in sorted(new_fp, key=lambda x: x["score"]):
        summary += f"- {r['stem'][:50]}  score={r['score']}  shade={r['shade']}\n"

    summary += "\n#### Lost flags — were chunky under SAM, now clean under YOLO ROI\n"
    for r in sorted(lost_tp, key=lambda x: x["score"], reverse=True):
        summary += f"- {r['stem'][:50]}  score={r['score']}  shade={r['shade']}\n"

    summary += f"""
## All flagged smoothies (worst first)
| rank | image | score | px | was (SAM) |
|---|---|---|---|---|
"""
    for rank, r in enumerate(flagged, 1):
        sam_v = r.get("sam_verdict", "?")
        note  = " **NEW**" if sam_v == "clean" else ""
        summary += f"| {rank} | {r['stem'][:28]} | {r['score']:.3f} | {r['chunk_pixels']} | {sam_v}{note} |\n"

    (out / "README.md").write_text(summary)
    print(f"\n{'='*60}")
    print(f"SAM baseline:        {sam_flag} flagged")
    print(f"YOLO {args.tag} result:      {n_flag} flagged  (Δ {n_flag - sam_flag:+d})")
    print(f"  New flags (FP?):   {len(new_fp)}")
    print(f"  Lost flags (miss): {len(lost_tp)}")
    print(f"{'='*60}")
    print(f"open → {out / 'flagged.png'}")
    print(f"       {out / 'README.md'}")


if __name__ == "__main__":
    main()
