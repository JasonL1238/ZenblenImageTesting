"""Compare YOLO-seg container detection vs cached SAM ROI masks across all 92 images.

SAM is the training-free reference detector — use this after training a new
YOLO model to sanity-check its masks before promoting the weights.

Outputs (under outputs/yolo_vs_sam/):
  comparison_grid.png   — side-by-side: original | SAM mask | YOLO mask | diff
  scores.csv            — per-image IoU, bbox overlap, YOLO confidence, which is bigger
  README.md             — plain-English summary

Usage:
  /opt/miniconda3/bin/python scripts/compare_yolo_vs_sam.py
  /opt/miniconda3/bin/python scripts/compare_yolo_vs_sam.py --weights runs/smoothie-seg/nano-v4/weights/best.pt
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

_ap = argparse.ArgumentParser()
_ap.add_argument("--weights", default="checkpoints/yolo_smoothie_seg.pt")
WEIGHTS = Path(_ap.parse_args().weights)
CACHE = Path("outputs/roi_cache_sam")
IMG_DIRS = [Path("data/images/red_pink"), Path("data/images/yellow")]
OUT = Path("outputs/yolo_vs_sam")
OUT.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR = OUT / "overlays"
OVERLAY_DIR.mkdir(exist_ok=True)

IMG_W = 220  # width of each panel in the grid


def load_sam_mask(stem: str, shape: tuple[int, int]) -> np.ndarray | None:
    p = CACHE / f"{stem}.png"
    if not p.exists():
        return None
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def yolo_mask_for_image(result, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Extract the best (highest-conf) mask from a YOLO result. Returns (mask, conf)."""
    h, w = shape
    if result.masks is None or len(result.masks) == 0:
        return np.zeros((h, w), dtype=np.uint8), 0.0

    confs = result.boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    conf = float(confs[best_idx])

    raw = result.masks.data[best_idx].cpu().numpy()  # float32 0..1
    mask = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > 0.5).astype(np.uint8), conf


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def colored_overlay(img: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.45) -> np.ndarray:
    out = img.copy()
    c = np.array(color, dtype=np.uint8)
    out[mask > 0] = (out[mask > 0] * (1 - alpha) + c * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def make_panel(img, mask, color, label, iou_val=None):
    vis = colored_overlay(img, mask, color)
    vis = cv2.resize(vis, (IMG_W, int(img.shape[0] * IMG_W / img.shape[1])))
    h_txt = 22
    bar = np.zeros((h_txt, vis.shape[1], 3), dtype=np.uint8)
    text = label if iou_val is None else f"{label}  IoU={iou_val:.3f}"
    cv2.putText(bar, text, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def diff_panel(img, sam_mask, yolo_mask):
    h, w = img.shape[:2]
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    both = np.logical_and(sam_mask, yolo_mask)
    sam_only = np.logical_and(sam_mask, ~yolo_mask.astype(bool))
    yolo_only = np.logical_and(yolo_mask, ~sam_mask.astype(bool))
    diff[both] = (50, 200, 50)       # green = agree
    diff[sam_only] = (255, 80, 80)    # blue  = SAM only (BGR)
    diff[yolo_only] = (80, 80, 255)   # red   = YOLO only
    vis = cv2.resize(diff, (IMG_W, int(h * IMG_W / w)))
    h_txt = 22
    bar = np.zeros((h_txt, vis.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, "diff  G=agree B=SAM R=YOLO", (4, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def main():
    model = YOLO(str(WEIGHTS))

    imgs = sorted(p for d in IMG_DIRS for p in d.glob("*.jpg"))
    print(f"Running YOLO on {len(imgs)} images …")

    rows = []
    panel_rows = []

    for i, img_path in enumerate(imgs, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [{i}] SKIP (unreadable): {img_path.name}")
            continue

        h, w = img.shape[:2]
        sam_mask = load_sam_mask(img_path.stem, (h, w))
        sam_ok = sam_mask is not None

        result = model(img, verbose=False)[0]
        yolo_mask, conf = yolo_mask_for_image(result, (h, w))
        yolo_ok = conf > 0

        iou_val = iou(sam_mask, yolo_mask) if (sam_ok and yolo_ok) else None
        sam_area = int(sam_mask.sum()) if sam_ok else 0
        yolo_area = int(yolo_mask.sum())

        bigger = "equal"
        if sam_area > yolo_area * 1.05:
            bigger = "SAM"
        elif yolo_area > sam_area * 1.05:
            bigger = "YOLO"

        rows.append({
            "stem": img_path.stem,
            "shade": img_path.parent.name,
            "iou": f"{iou_val:.4f}" if iou_val is not None else "N/A",
            "yolo_conf": f"{conf:.3f}",
            "yolo_detected": yolo_ok,
            "sam_cached": sam_ok,
            "sam_area_px": sam_area,
            "yolo_area_px": yolo_area,
            "bigger": bigger,
        })

        status = f"IoU={iou_val:.3f}" if iou_val is not None else "NO-DET"
        print(f"  [{i:2d}/{len(imgs)}] {img_path.stem[:40]}  {status}  conf={conf:.2f}  bigger={bigger}")

        # per-image overlay saved regardless
        sm = sam_mask if sam_ok else np.zeros((h, w), dtype=np.uint8)
        ym = yolo_mask

        p_orig = cv2.resize(img, (IMG_W, int(h * IMG_W / w)))
        bar = np.zeros((22, p_orig.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, img_path.stem[:32], (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        p_orig = np.vstack([bar, p_orig])

        p_sam  = make_panel(img, sm, (255, 100, 30), "SAM")
        p_yolo = make_panel(img, ym, (30, 180, 255), "YOLO", iou_val)
        p_diff = diff_panel(img, sm, ym)

        # pad to same height
        target_h = max(p.shape[0] for p in [p_orig, p_sam, p_yolo, p_diff])
        def pad_h(p):
            if p.shape[0] < target_h:
                pad = np.zeros((target_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
                return np.vstack([p, pad])
            return p

        row_img = np.hstack([pad_h(p) for p in [p_orig, p_sam, p_yolo, p_diff]])
        cv2.imwrite(str(OVERLAY_DIR / f"{img_path.stem}.png"), row_img)
        panel_rows.append(row_img)

    # write CSV
    csv_path = OUT / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # summary stats
    iou_vals = [float(r["iou"]) for r in rows if r["iou"] != "N/A"]
    no_det = sum(1 for r in rows if not r["yolo_detected"])
    low_iou = [r for r in rows if r["iou"] != "N/A" and float(r["iou"]) < 0.70]
    good_iou = [r for r in rows if r["iou"] != "N/A" and float(r["iou"]) >= 0.90]
    bigger_sam  = sum(1 for r in rows if r["bigger"] == "SAM")
    bigger_yolo = sum(1 for r in rows if r["bigger"] == "YOLO")

    summary = f"""# YOLO vs SAM container detection comparison
## Dataset: {len(rows)} images ({sum(1 for r in rows if r['shade']=='red_pink')} red/pink, {sum(1 for r in rows if r['shade']=='yellow')} yellow)
## Model: {WEIGHTS}

### Overall
- YOLO detected container: {len(rows) - no_det}/{len(rows)} images
- YOLO missed (no detection): {no_det}
- Mean IoU (where both detected): {sum(iou_vals)/len(iou_vals):.3f}
- Median IoU: {float(np.median(iou_vals)):.3f}
- Min IoU: {min(iou_vals):.3f}
- Max IoU: {max(iou_vals):.3f}

### Agreement buckets
- IoU ≥ 0.90 (good agreement): {len(good_iou)}
- IoU 0.70–0.90 (moderate): {len([r for r in rows if r['iou'] != 'N/A' and 0.70 <= float(r['iou']) < 0.90])}
- IoU < 0.70 (poor): {len(low_iou)}

### Size comparison (5% tolerance)
- SAM bigger: {bigger_sam}
- YOLO bigger: {bigger_yolo}
- Roughly equal: {len(rows) - bigger_sam - bigger_yolo}

### Low-IoU images (< 0.70)
"""
    for r in sorted(low_iou, key=lambda x: float(x["iou"])):
        summary += f"- {r['stem'][:45]}  IoU={r['iou']}  shade={r['shade']}  bigger={r['bigger']}\n"

    readme_path = OUT / "README.md"
    readme_path.write_text(summary)
    print("\n" + summary)

    # build grid: tile panel rows into a big image (max 30 rows to keep file size sane)
    GRID_MAX = 92
    grid_rows = panel_rows[:GRID_MAX]
    # unify widths
    max_w = max(r.shape[1] for r in grid_rows)
    padded = []
    for r in grid_rows:
        if r.shape[1] < max_w:
            pad = np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)
            padded.append(np.hstack([r, pad]))
        else:
            padded.append(r)
    grid = np.vstack(padded)
    grid_path = OUT / "comparison_grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"Grid saved → {grid_path}  ({grid.shape[1]}×{grid.shape[0]})")
    print(f"CSV  saved → {csv_path}")
    print(f"Per-image overlays → {OVERLAY_DIR}/")


if __name__ == "__main__":
    main()
