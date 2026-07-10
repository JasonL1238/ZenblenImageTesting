"""Run the FULL pipeline (YOLO ROI live + classical chunk detector) on a fresh,
labeling-disjoint test folder and write a review report.

Adapted from scripts/validate_chunks.py (the 92-image validator) for an arbitrary
flat folder of UserGrab_*.jpg. Differences:
  - --images points at any folder (default outputs/pipeline_test_set/images), globbed recursively.
  - ROI is always run LIVE via YOLO (no cache — this is a never-seen set).
  - shade is classified per-image (run.py's _classify_smoothie) for stratified analysis.
  - ROI failures (empty mask) are tracked as a SEPARATE verdict ("no_roi"), not
    silently scored 1.0 — ROI robustness is a headline metric for this eval.

Outputs (under --out, default outputs/pipeline_test_set/report/):
  overlays/<stem>.png  — triptych [ original | ROI | chunk detection ]
  flagged.png          — flagged smoothies, worst-first
  scores.csv           — stem, shade, score, chunk_pixels, verdict, roi_pixels, roi_ok
  README.md            — summary counts + score distribution
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
from smoothie_cv.detection import _classify_smoothie, SmoothieType

FLAG_SCORE    = 0.999
PAIR_W        = 230
PAIRS_PER_ROW = 3


def _shade(img) -> str:
    try:
        return "red_pink" if _classify_smoothie(img) == SmoothieType.RED_PINK else "yellow"
    except Exception:
        return "unknown"


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
    roi_vis = _label(_roi_strip(img, roi_mask), ["ROI mask"], (0, 255, 0))
    det     = overlay_mask(img, chunk_mask, color=(255, 0, 0), alpha=0.55)
    verdict = "CHUNKS" if score < FLAG_SCORE else "clean"
    det     = _label(det, [f"detect: {verdict}", f"blend {score:.3f}  {px}px"], (80, 160, 255))
    gap     = np.full((img.shape[0], 6, 3), 40, np.uint8)
    return np.hstack([orig, gap, roi_vis, gap, det])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="outputs/pipeline_test_set/images")
    ap.add_argument("--weights", default="checkpoints/yolo_smoothie_seg.pt")
    ap.add_argument("--out", default="outputs/pipeline_test_set/report")
    ap.add_argument("--limit", type=int, default=0, help="cap #images (0=all), for a quick smoke run")
    ap.add_argument("--roi-cache", default=None,
                    help="dir of cached <stem>.png ROI masks; skips live YOLO ROI "
                         "(both A/B runs then share identical ROIs)")
    ap.add_argument("--logo-yolo", action="store_true",
                    help="enable trained-logo-mask chunk suppression (runs logo model live)")
    ap.add_argument("--logo-weights", default=None,
                    help="logo YOLO weights (default config.logo_weights)")
    args = ap.parse_args()

    out      = Path(args.out)
    overlays = out / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)

    roi_cache = Path(args.roi_cache) if args.roi_cache else None
    model = None
    if roi_cache is None:
        from ultralytics import YOLO
        weights = Path(args.weights)
        if not weights.exists():
            print(f"ERROR: weights not found: {weights}")
            sys.exit(1)
        print(f"Loading YOLO ({weights}) …")
        model = YOLO(str(weights))
    elif not roi_cache.is_dir():
        print(f"ERROR: ROI cache not found: {roi_cache}")
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
    from smoothie_cv.detection.yolo import get_yolo_roi

    imgs = sorted(Path(args.images).rglob("*.jpg"))
    if args.limit:
        imgs = imgs[:args.limit]
    print(f"{len(imgs)} images from {args.images}")

    rows      = []
    originals = {}

    for i, p in enumerate(imgs, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"  [{i}] UNREADABLE {p.name}")
            continue
        h, w = img.shape[:2]
        shade = _shade(img)

        try:
            if roi_cache is not None:
                cached = cv2.imread(str(roi_cache / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
                roi = cached if cached is not None else np.zeros((h, w), dtype=np.uint8)
            else:
                roi = get_yolo_roi(model(img, verbose=False)[0], (h, w))
        except Exception as e:
            print(f"  [{i}] ROI ERROR {p.stem[:36]}: {e}")
            roi = np.zeros((h, w), dtype=np.uint8)

        roi_px = int((roi > 0).sum())
        if roi_px == 0:
            rows.append({"stem": p.stem, "shade": shade, "score": "",
                         "chunk_pixels": 0, "verdict": "no_roi",
                         "roi_pixels": 0, "roi_ok": False})
            print(f"  [{i}/{len(imgs)}] NO_ROI  {p.stem[:40]}")
            # still write an overlay so we can eyeball the failure
            cv2.imwrite(str(overlays / f"{p.stem}.png"),
                        _label(img.copy(), ["NO ROI — YOLO found no container"], (0, 0, 255)))
            continue

        r  = pipe.analyze(img, roi)
        px = int((r.mask > 0).sum())
        verdict = "chunks" if r.blend_score < FLAG_SCORE else "clean"

        rows.append({"stem": p.stem, "shade": shade,
                     "score": round(r.blend_score, 4), "chunk_pixels": px,
                     "verdict": verdict, "roi_pixels": roi_px, "roi_ok": True})
        if i % 20 == 0 or verdict == "chunks":
            print(f"  [{i}/{len(imgs)}] {verdict:6s} score={r.blend_score:.3f} roi={roi_px} {p.stem[:32]}")

        cv2.imwrite(str(overlays / f"{p.stem}.png"),
                    _triptych(img, roi, r.mask, r.blend_score, px))
        originals[p.stem] = (img, overlay_mask(img, r.mask, color=(255, 0, 0), alpha=0.55),
                             r.blend_score, px)

    # ── flagged.png ──
    flagged = sorted([r for r in rows if r["verdict"] == "chunks"], key=lambda r: r["score"])
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
        orig_f = _label(orig_f, [f"#{rank}  {short}", "original"], (255, 255, 255))
        over_f = _label(over_f, [f"blend {score:.3f}", f"{px}px detected"], (80, 160, 255))
        cells.append(np.hstack([orig_f, over_f]))

    if cells:
        ch    = max(c.shape[0] for c in cells)
        cells = [cv2.copyMakeBorder(c, 0, ch - c.shape[0], 0, 0, cv2.BORDER_CONSTANT) for c in cells]
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
        writer = csv.DictWriter(f, fieldnames=["stem", "shade", "score", "chunk_pixels",
                                               "verdict", "roi_pixels", "roi_ok"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["verdict"], str(r["score"]))))

    # ── README.md summary + score histogram ──
    n_total  = len(rows)
    n_chunks = sum(1 for r in rows if r["verdict"] == "chunks")
    n_clean  = sum(1 for r in rows if r["verdict"] == "clean")
    n_noroi  = sum(1 for r in rows if r["verdict"] == "no_roi")
    by_shade = {}
    for r in rows:
        by_shade.setdefault(r["shade"], {"clean": 0, "chunks": 0, "no_roi": 0})
        by_shade[r["shade"]][r["verdict"]] += 1
    scored = [r["score"] for r in rows if r["roi_ok"]]
    # score histogram buckets
    buckets = {"1.000 (clean)": 0, "0.99-1.0": 0, "0.95-0.99": 0, "0.90-0.95": 0, "<0.90": 0}
    for s in scored:
        if s >= 0.9995: buckets["1.000 (clean)"] += 1
        elif s >= 0.99: buckets["0.99-1.0"] += 1
        elif s >= 0.95: buckets["0.95-0.99"] += 1
        elif s >= 0.90: buckets["0.90-0.95"] += 1
        else: buckets["<0.90"] += 1

    md = [f"# Full-pipeline test — {n_total} fresh images (labeling-disjoint)",
          f"ROI: live YOLO ({weights})  ·  chunks: classical deviation",
          "",
          f"- **clean**: {n_clean}",
          f"- **chunks (flagged)**: {n_chunks}",
          f"- **NO ROI (detector failed)**: {n_noroi}",
          "",
          "## By shade",
          "| shade | clean | chunks | no_roi |",
          "|---|---|---|---|"]
    for sh, c in sorted(by_shade.items()):
        md.append(f"| {sh} | {c['clean']} | {c['chunks']} | {c['no_roi']} |")
    md += ["", "## Blend-score distribution (ROI-ok images)",
           "| bucket | count |", "|---|---|"]
    for b, c in buckets.items():
        md.append(f"| {b} | {c} |")
    md += ["", "## All flagged (worst first)", "| rank | image | score | px | shade |", "|---|---|---|---|---|"]
    for rank, r in enumerate(flagged, 1):
        md.append(f"| {rank} | {r['stem'][:30]} | {r['score']} | {r['chunk_pixels']} | {r['shade']} |")
    (out / "README.md").write_text("\n".join(md))

    print(f"\n{'='*60}")
    print(f"total {n_total}: {n_clean} clean, {n_chunks} chunks, {n_noroi} NO_ROI")
    print(f"open → {out/'flagged.png'}\n       {out/'README.md'}")


if __name__ == "__main__":
    main()
