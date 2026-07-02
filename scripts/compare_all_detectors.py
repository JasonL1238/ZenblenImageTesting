"""Three-way container detection comparison: SAM vs YOLO nano-v2 vs YOLO nano-v3.

Outputs (under outputs/detector_comparison/):
  comparison_grid.png   — all 92 images stacked: original | SAM | v2 | v3 | diff(v2→v3)
  scores.csv            — per-image IoU vs SAM for each model + size comparison
  README.md             — summary stats

Usage:
  /opt/miniconda3/bin/python scripts/compare_all_detectors.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WEIGHTS_V2 = Path("runs/smoothie-seg/nano-v2/weights/best.pt")
WEIGHTS_V3 = Path("runs/smoothie-seg/nano-v3/weights/best.pt")
CACHE       = Path("outputs/roi_cache")
IMG_DIRS    = [Path("data/images/red_pink"), Path("data/images/yellow")]
OUT         = Path("outputs/detector_comparison")
OUT.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR = OUT / "overlays"
OVERLAY_DIR.mkdir(exist_ok=True)

PANEL_W = 200  # width of each panel in the side-by-side


def load_sam(stem: str, shape: tuple[int, int]) -> np.ndarray:
    p = CACHE / f"{stem}.png"
    if not p.exists():
        return np.zeros(shape, dtype=np.uint8)
    m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(shape, dtype=np.uint8)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 0).astype(np.uint8)


def best_mask(result, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    h, w = shape
    if result.masks is None or len(result.masks) == 0:
        return np.zeros((h, w), dtype=np.uint8), 0.0
    confs = result.boxes.conf.cpu().numpy()
    idx = int(np.argmax(confs))
    raw = result.masks.data[idx].cpu().numpy()
    m = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m > 0.5).astype(np.uint8), float(confs[idx])


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def overlay(img: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.4) -> np.ndarray:
    out = img.copy()
    out[mask > 0] = (out[mask > 0] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def labeled_panel(img: np.ndarray, mask: np.ndarray, color: tuple, label: str) -> np.ndarray:
    vis = overlay(img, mask, color)
    h, w = img.shape[:2]
    vis = cv2.resize(vis, (PANEL_W, int(h * PANEL_W / w)))
    bar = np.zeros((20, vis.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def diff_v2_v3(img: np.ndarray, m_sam: np.ndarray, m_v2: np.ndarray, m_v3: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[np.logical_and(m_v2, m_v3)] = (50, 200, 50)         # green  = both agree
    diff[np.logical_and(m_v2, ~m_v3.astype(bool))] = (255, 80, 80)   # blue   = v2 only
    diff[np.logical_and(m_v3, ~m_v2.astype(bool))] = (80, 80, 255)   # red    = v3 only
    vis = cv2.resize(diff, (PANEL_W, int(h * PANEL_W / w)))
    bar = np.zeros((20, vis.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, "v2 vs v3  G=both B=v2 R=v3", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def pad_height(panel: np.ndarray, target_h: int) -> np.ndarray:
    if panel.shape[0] < target_h:
        pad = np.zeros((target_h - panel.shape[0], panel.shape[1], 3), dtype=np.uint8)
        return np.vstack([panel, pad])
    return panel


def main() -> None:
    for wpath, label in [(WEIGHTS_V2, "v2"), (WEIGHTS_V3, "v3")]:
        if not wpath.exists():
            print(f"ERROR: weights not found: {wpath}")
            print("Make sure both nano-v2 and nano-v3 runs are complete.")
            sys.exit(1)

    print("Loading models …")
    model_v2 = YOLO(str(WEIGHTS_V2))
    model_v3 = YOLO(str(WEIGHTS_V3))

    imgs = sorted(p for d in IMG_DIRS for p in d.glob("*.jpg"))
    print(f"Running inference on {len(imgs)} images …")

    rows = []
    panel_rows = []

    for i, img_path in enumerate(imgs, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        shape = (h, w)

        sam  = load_sam(img_path.stem, shape)
        m_v2, conf_v2 = best_mask(model_v2(img, verbose=False)[0], shape)
        m_v3, conf_v3 = best_mask(model_v3(img, verbose=False)[0], shape)

        iou_v2 = iou(sam, m_v2)
        iou_v3 = iou(sam, m_v3)
        iou_v2_v3 = iou(m_v2, m_v3)

        rows.append({
            "stem":    img_path.stem,
            "shade":   img_path.parent.name,
            "iou_v2_vs_sam":   f"{iou_v2:.4f}",
            "iou_v3_vs_sam":   f"{iou_v3:.4f}",
            "iou_v2_vs_v3":    f"{iou_v2_v3:.4f}",
            "conf_v2": f"{conf_v2:.3f}",
            "conf_v3": f"{conf_v3:.3f}",
            "v3_improved": iou_v3 > iou_v2 + 0.01,
            "v3_regressed": iou_v2 > iou_v3 + 0.01,
        })

        tag = ""
        if iou_v3 > iou_v2 + 0.01:  tag = "↑"
        elif iou_v2 > iou_v3 + 0.01: tag = "↓"
        print(f"  [{i:2d}/{len(imgs)}] v2={iou_v2:.3f}  v3={iou_v3:.3f} {tag}  v2↔v3={iou_v2_v3:.3f}")

        # build 5-panel row: original | SAM | v2 | v3 | v2-vs-v3 diff
        orig = cv2.resize(img, (PANEL_W, int(h * PANEL_W / w)))
        bar  = np.zeros((20, orig.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, img_path.stem[:30], (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (200, 200, 200), 1, cv2.LINE_AA)
        p_orig = np.vstack([bar, orig])
        p_sam  = labeled_panel(img, sam,  (255, 140, 0),   "SAM")
        p_v2   = labeled_panel(img, m_v2, (30,  200, 255), f"v2  IoU={iou_v2:.3f}")
        p_v3   = labeled_panel(img, m_v3, (80,  255, 80),  f"v3  IoU={iou_v3:.3f}")
        p_diff = diff_v2_v3(img, sam, m_v2, m_v3)

        th = max(p.shape[0] for p in [p_orig, p_sam, p_v2, p_v3, p_diff])
        row = np.hstack([pad_height(p, th) for p in [p_orig, p_sam, p_v2, p_v3, p_diff]])
        cv2.imwrite(str(OVERLAY_DIR / f"{img_path.stem}.png"), row)
        panel_rows.append(row)

    # CSV
    with open(OUT / "scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # summary
    v2_ious = [float(r["iou_v2_vs_sam"]) for r in rows]
    v3_ious = [float(r["iou_v3_vs_sam"]) for r in rows]
    improved  = sum(1 for r in rows if r["v3_improved"])
    regressed = sum(1 for r in rows if r["v3_regressed"])

    summary = f"""# Three-way detector comparison: SAM vs nano-v2 vs nano-v3
## {len(rows)} images ({sum(1 for r in rows if r['shade']=='red_pink')} red/pink, {sum(1 for r in rows if r['shade']=='yellow')} yellow)

|                  | nano-v2 | nano-v3 |
|------------------|---------|---------|
| Mean IoU vs SAM  | {sum(v2_ious)/len(v2_ious):.3f}   | {sum(v3_ious)/len(v3_ious):.3f}   |
| Median IoU vs SAM| {float(np.median(v2_ious)):.3f}   | {float(np.median(v3_ious)):.3f}   |
| Min IoU vs SAM   | {min(v2_ious):.3f}   | {min(v3_ious):.3f}   |
| IoU ≥ 0.90       | {sum(1 for x in v2_ious if x >= 0.90)}       | {sum(1 for x in v3_ious if x >= 0.90)}       |
| IoU < 0.70       | {sum(1 for x in v2_ious if x < 0.70)}       | {sum(1 for x in v3_ious if x < 0.70)}       |

### v3 vs v2 (> 0.01 IoU change)
- Improved:  {improved}
- Regressed: {regressed}
- Same:      {len(rows) - improved - regressed}

### Images where v3 regressed vs v2
"""
    for r in sorted([r for r in rows if r["v3_regressed"]], key=lambda x: float(x["iou_v3_vs_sam"])):
        summary += f"- {r['stem'][:50]}  v2={r['iou_v2_vs_sam']}  v3={r['iou_v3_vs_sam']}  shade={r['shade']}\n"

    (OUT / "README.md").write_text(summary)
    print("\n" + summary)

    # stitch grid
    max_w = max(r.shape[1] for r in panel_rows)
    grid = np.vstack([
        np.hstack([r, np.zeros((r.shape[0], max_w - r.shape[1], 3), dtype=np.uint8)])
        if r.shape[1] < max_w else r
        for r in panel_rows
    ])
    grid_path = OUT / "comparison_grid.png"
    cv2.imwrite(str(grid_path), grid)
    print(f"Grid   → {grid_path}  ({grid.shape[1]}×{grid.shape[0]})")
    print(f"CSV    → {OUT}/scores.csv")
    print(f"Per-image → {OVERLAY_DIR}/")


if __name__ == "__main__":
    main()
