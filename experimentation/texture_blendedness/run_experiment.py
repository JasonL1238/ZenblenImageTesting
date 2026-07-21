"""
Texture-blendedness experiment runner.

Samples images from training/labeling/data/images/, runs:
  - YOLO standard-seg  -> liquid ROI
  - YOLO logo-seg      -> logo mask (excluded from ROI)
then computes the chunk-INDEPENDENT texture-blendedness score and writes
per-image overlays + a scores.csv for manual review.

No chunk detection, no label filtering — labels are irrelevant to this metric.

Run:
  /opt/miniconda3/bin/python experimentation/texture_blendedness/run_experiment.py --n 100
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from texture import (  # noqa: E402
    TextureParams,
    compute_texture_score,
    outline_deviation,
    overlay_deviation,
)

REPO = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO / "training" / "labeling" / "data" / "images"
STD_WEIGHTS = REPO / "training" / "checkpoints" / "yolo_standard_seg.pt"
LOGO_WEIGHTS = REPO / "training" / "checkpoints" / "yolo_logo_seg.pt"
OUT_DIR = Path(__file__).resolve().parent / "outputs"

_STD_MODEL = None
_LOGO_MODEL = None


def _load_models():
    global _STD_MODEL, _LOGO_MODEL
    from ultralytics import YOLO

    for w in (STD_WEIGHTS, LOGO_WEIGHTS):
        if not w.exists():
            raise FileNotFoundError(f"Missing weights: {w}")
    _STD_MODEL = YOLO(str(STD_WEIGHTS))
    _LOGO_MODEL = YOLO(str(LOGO_WEIGHTS))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ff = mask.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return mask | cv2.bitwise_not(ff)


def _roi_mask(image: np.ndarray) -> np.ndarray:
    """Highest-confidence smoothie instance -> hole-filled 255 mask."""
    h, w = image.shape[:2]
    res = _STD_MODEL(image, verbose=False, device="cpu")[0]
    if res.masks is None or len(res.masks) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    confs = res.boxes.conf.cpu().numpy()
    raw = res.masks.data[int(np.argmax(confs))].cpu().numpy()
    m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return _fill_holes(((m > 0.5) * 255).astype(np.uint8))


def _logo_mask(image: np.ndarray) -> np.ndarray:
    """Union of all logo instances -> 255 mask (no hole-fill)."""
    h, w = image.shape[:2]
    res = _LOGO_MODEL(image, verbose=False, device="cpu", conf=0.25)[0]
    mask = np.zeros((h, w), dtype=np.uint8)
    if res.masks is None or len(res.masks) == 0:
        return mask
    for raw in res.masks.data.cpu().numpy():
        m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[m > 0.5] = 255
    return mask


def _sample_images(n: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in exts)
    if not files:
        raise FileNotFoundError(f"No images under {IMAGES_DIR}")
    if len(files) <= n:
        return files
    # deterministic even spread across the whole set (no RNG -> reproducible)
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="number of images")
    ap.add_argument("--blur", type=int, default=121)
    ap.add_argument("--thresh", type=float, default=6.0)
    ap.add_argument("--open", type=int, default=7)
    ap.add_argument("--demax", type=float, default=25.0)
    ap.add_argument("--agg", choices=["area", "sev", "mean"], default="area")
    ap.add_argument("--strictness", type=float, default=2.0)
    ap.add_argument("--overlay", choices=["outline", "heat", "sidebyside"],
                    default="outline",
                    help="outline=see-through contours (default), "
                         "heat=filled heatmap, sidebyside=clean | heatmap")
    args = ap.parse_args()

    params = TextureParams(
        blur_kernel=args.blur,
        delta_e_thresh=args.thresh,
        open_kernel_px=args.open,
        delta_e_max=args.demax,
        agg=args.agg,
        strictness=args.strictness,
    )

    _load_models()
    images = _sample_images(args.n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overlays_dir = OUT_DIR / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    rows = []
    print(f"Running {len(images)} images  (params: {params})")
    for i, path in enumerate(images, 1):
        image = cv2.imread(str(path))
        if image is None:
            print(f"  [{i}] SKIP unreadable {path.name}")
            continue

        roi = _roi_mask(image)
        logo = _logo_mask(image)
        roi_px = int((roi > 0).sum())

        if roi_px == 0:
            score, dev = 100.0, np.zeros(image.shape[:2], np.float32)
            note = "no_roi"
        else:
            score, dev = compute_texture_score(image, roi, logo, params)
            note = "ok"

        # build the review visual per chosen mode
        roi_cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if args.overlay == "heat":
            vis = overlay_deviation(image, dev, params.delta_e_max)
            cv2.drawContours(vis, roi_cnts, -1, (0, 255, 0), 2)
        elif args.overlay == "sidebyside":
            left = image.copy()  # untouched original
            right = outline_deviation(image, dev, mild_thresh=params.delta_e_thresh * 1.5)
            cv2.drawContours(right, roi_cnts, -1, (0, 255, 0), 1)
            vis = np.hstack([left, right])
        else:  # outline (default): see-through contours, image stays visible
            vis = outline_deviation(image, dev, mild_thresh=params.delta_e_thresh * 1.5)
            cv2.drawContours(vis, roi_cnts, -1, (0, 255, 0), 1)
        cv2.putText(vis, f"{score:.1f}", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(vis, f"{score:.1f}", (12, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1)
        # temp name during loop; renamed with a score-rank prefix after sorting
        tmp_name = f"_tmp_{i:03d}.jpg"
        cv2.imwrite(str(overlays_dir / tmp_name), vis)

        rows.append({
            "idx": i,
            "score": round(score, 2),
            "roi_px": roi_px,
            "logo_px": int((logo > 0).sum()),
            "note": note,
            "source": str(path.relative_to(REPO)),
            "_tmp": tmp_name,
            "stem": path.stem,
        })
        print(f"  [{i}/{len(images)}] {score:6.1f}  {note:6}  {path.name}")

    # rank ascending by score so the LEAST blended sort to the top of the folder
    rows.sort(key=lambda r: r["score"])
    for rank, r in enumerate(rows, 1):
        out_name = f"{rank:03d}_score{r['score']:05.1f}_{r['stem']}.jpg"
        (overlays_dir / r.pop("_tmp")).rename(overlays_dir / out_name)
        r["rank"] = rank
        r["overlay"] = out_name
        del r["stem"]

    with open(OUT_DIR / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rank", "score", "idx", "roi_px", "logo_px", "note", "source", "overlay"])
        writer.writeheader()
        writer.writerows(rows)

    scored = [r["score"] for r in rows if r["note"] == "ok"]
    print(f"\nDone. {len(rows)} images, {len(scored)} scored, "
          f"{sum(1 for r in rows if r['note'] == 'no_roi')} no-ROI.")
    if scored:
        arr = np.array(scored)
        print(f"score  min={arr.min():.1f}  med={np.median(arr):.1f}  "
              f"max={arr.max():.1f}  mean={arr.mean():.1f}")
    print(f"Review: {OUT_DIR/'scores.csv'}  and  {overlays_dir}/")


if __name__ == "__main__":
    main()
