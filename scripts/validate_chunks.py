"""Run the chunk detector on every image (using cached SAM ROIs) and write a
clean, human-readable report.

Usage: python scripts/validate_chunks.py

Outputs (all under outputs/report/):
  flagged.png        ONE readable view: only smoothies where chunks were found,
                     original vs. detection side-by-side, sorted worst-first.
  overlays/<stem>.png  per-image detection overlay (drill-down, all images).
  scores.csv         stem, score, chunk_pixels, verdict for every image.
  README.md          plain-English summary.

Cached ROIs come from outputs/roi_cache/ (build with scripts/cache_sam_rois.py).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoothie_cv.config import Config
from smoothie_cv.pipelines.classical_cv import ClassicalCVPipeline
from smoothie_cv.scoring.metrics import overlay_mask

CACHE = Path("outputs/roi_cache")
OUT = Path("outputs/report")
OVERLAYS = OUT / "overlays"

# A smoothie is "flagged" (chunks found) below this blend score. 1.0 = perfectly blended.
FLAG_SCORE = 0.999
PAIR_W = 230          # width of each image in the side-by-side pair
PAIRS_PER_ROW = 3


def _label(img, lines, color):
    cv2.rectangle(img, (0, 0), (img.shape[1], 16 + 18 * len(lines)), (0, 0, 0), -1)
    for i, t in enumerate(lines):
        cv2.putText(img, t, (4, 16 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img


def main() -> None:
    OVERLAYS.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    pipe = ClassicalCVPipeline(cfg)
    imgs = sorted(Path("data/images").rglob("*.jpg"))

    rows = []
    originals = {}
    for p in imgs:
        roi_path = CACHE / f"{p.stem}.png"
        if not roi_path.exists():
            continue
        img = cv2.imread(str(p))
        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        r = pipe.analyze(img, roi)
        px = int((r.mask > 0).sum())
        verdict = "chunks" if r.blend_score < FLAG_SCORE else "clean"
        rows.append({"stem": p.stem, "score": round(r.blend_score, 4),
                     "chunk_pixels": px, "verdict": verdict})
        vis = overlay_mask(img, r.mask, color=(255, 0, 0), alpha=0.55)
        cv2.imwrite(str(OVERLAYS / f"{p.stem}.png"), vis)
        originals[p.stem] = (img, vis, r.blend_score, px)

    # ── one readable view: only the flagged smoothies, sorted worst-first ──
    flagged = sorted([r for r in rows if r["verdict"] == "chunks"], key=lambda r: r["score"])
    cells = []
    for rank, r in enumerate(flagged, 1):
        img, vis, score, px = originals[r["stem"]]

        def fit(im):
            h, w = im.shape[:2]
            return cv2.resize(im, (PAIR_W, int(h * PAIR_W / w)))

        orig, over = fit(img), fit(vis)
        short = r["stem"].replace("UserGrab_", "")[:8]
        orig = _label(orig, [f"#{rank}  {short}", "original"], (255, 255, 255))
        over = _label(over, [f"blend {score:.3f}", f"{px}px detected"], (80, 160, 255))
        cells.append(np.hstack([orig, over]))

    if cells:
        ch = max(c.shape[0] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in cells]
        gap = 8
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
        cv2.imwrite(str(OUT / "flagged.png"), montage)

    # ── scores.csv ──
    with open(OUT / "scores.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["stem", "score", "chunk_pixels", "verdict"])
        wr.writeheader()
        wr.writerows(sorted(rows, key=lambda r: r["score"]))

    # ── README.md ──
    n_flag = len(flagged)
    n_total = len(rows)
    lines = [
        "# Chunk-detection report\n",
        f"**{n_flag} of {n_total}** smoothies were flagged for unblended chunks; "
        f"the other **{n_total - n_flag}** scored clean (well blended).\n",
        "- `flagged.png` — the flagged smoothies at a glance: original (left) vs. "
        "detection (right, chunks highlighted in blue), worst first.",
        "- `scores.csv` — blend score (1.0 = perfectly blended) for every image.",
        "- `overlays/` — the detection overlay for every image, for drill-down.\n",
        "## Flagged smoothies (worst first)\n",
        "| rank | image | blend score | pixels |",
        "|---|---|---|---|",
    ]
    for rank, r in enumerate(flagged, 1):
        lines.append(f"| {rank} | {r['stem'][:24]} | {r['score']:.3f} | {r['chunk_pixels']} |")
    (OUT / "README.md").write_text("\n".join(lines) + "\n")

    print(f"{n_total} images · {n_flag} flagged · {n_total - n_flag} clean")
    print(f"open → {OUT/'flagged.png'}  (and {OUT/'README.md'})")


if __name__ == "__main__":
    main()
