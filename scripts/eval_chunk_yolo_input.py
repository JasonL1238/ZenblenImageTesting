#!/usr/bin/env python3
"""A/B: chunk YOLO on ROI crop vs full-frame + ROI filter.

Compares two inference modes on labeling/chunk_dataset val+test (held-out
polygon labels), using YOLO-standard for the smoothie ROI:

  roi_crop     — crop to ROI, run chunk YOLO on the crop, AND with ROI
  full_filter  — run chunk YOLO on the full frame, keep pixels inside ROI

Reports mean/median mask IoU, pixel P/R/F1, image-level flagged agreement,
writes overlays under outputs/chunk_yolo_input_ab/, and prints the recommended
config.chunk_yolo_input winner.

Usage:
  python scripts/eval_chunk_yolo_input.py
  python scripts/eval_chunk_yolo_input.py --splits val test --out outputs/chunk_yolo_input_ab
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from smoothie_cv.config import Config
from smoothie_cv.detection import detect_container
from smoothie_cv.detection.chunk import detect_chunk
from smoothie_cv.scoring.metrics import overlay_mask

DATASET = REPO / "labeling" / "chunk_dataset"
MODES = ("full_filter", "roi_crop")


def yolo_seg_label_to_mask(label_path: Path, h: int, w: int) -> np.ndarray:
    """Rasterize YOLO-seg polygon .txt (class + normalized xy pairs) to uint8 mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if not label_path.exists():
        return mask
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least 3 points
            continue
        coords = list(map(float, parts[1:]))
        pts = np.array(
            [[int(round(coords[i] * w)), int(round(coords[i + 1] * h))]
             for i in range(0, len(coords), 2)],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            cv2.fillPoly(mask, [pts], 255)
    return mask


def iter_split_pairs(dataset: Path, splits: list[str]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for split in splits:
        img_dir = dataset / "images" / split
        lbl_dir = dataset / "labels" / split
        if not img_dir.is_dir():
            continue
        for img_path in sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png")):
            lbl = lbl_dir / f"{img_path.stem}.txt"
            pairs.append((img_path, lbl))
    return pairs


def mask_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    p = pred > 0
    g = gt > 0
    inter = int(np.logical_and(p, g).sum())
    union = int(np.logical_or(p, g).sum())
    tp = inter
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    iou = tp / union if union else (1.0 if tp == 0 and fp == 0 and fn == 0 else 0.0)
    prec = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    rec = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "iou": iou, "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "pred_flag": bool(p.any()), "gt_flag": bool(g.any()),
    }


def summarize(rows: list[dict], prefix: str) -> dict[str, float]:
    ious = [r[f"{prefix}_iou"] for r in rows]
    precs = [r[f"{prefix}_precision"] for r in rows]
    recs = [r[f"{prefix}_recall"] for r in rows]
    f1s = [r[f"{prefix}_f1"] for r in rows]
    agree = sum(1 for r in rows if r[f"{prefix}_pred_flag"] == r["gt_flag"])
    tp_img = sum(1 for r in rows if r[f"{prefix}_pred_flag"] and r["gt_flag"])
    fp_img = sum(1 for r in rows if r[f"{prefix}_pred_flag"] and not r["gt_flag"])
    fn_img = sum(1 for r in rows if not r[f"{prefix}_pred_flag"] and r["gt_flag"])
    return {
        "mean_iou": float(np.mean(ious)),
        "median_iou": float(np.median(ious)),
        "mean_precision": float(np.mean(precs)),
        "mean_recall": float(np.mean(recs)),
        "mean_f1": float(np.mean(f1s)),
        "img_agree": agree / len(rows) if rows else 0.0,
        "img_tp": tp_img,
        "img_fp": fp_img,
        "img_fn": fn_img,
        "n": len(rows),
    }


def pick_winner(stats: dict[str, dict[str, float]]) -> str:
    """Mean IoU primary; image-level precision then recall as tie-break."""
    a, b = MODES
    sa, sb = stats[a], stats[b]
    if abs(sa["mean_iou"] - sb["mean_iou"]) > 1e-6:
        return a if sa["mean_iou"] > sb["mean_iou"] else b
    # image precision = tp / (tp+fp)
    def img_prec(s: dict) -> float:
        d = s["img_tp"] + s["img_fp"]
        return s["img_tp"] / d if d else 0.0

    def img_rec(s: dict) -> float:
        d = s["img_tp"] + s["img_fn"]
        return s["img_tp"] / d if d else 0.0

    pa, pb = img_prec(sa), img_prec(sb)
    if abs(pa - pb) > 1e-9:
        return a if pa > pb else b
    ra, rb = img_rec(sa), img_rec(sb)
    if abs(ra - rb) > 1e-9:
        return a if ra > rb else b
    return a  # prefer full_filter on exact tie (matches training)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--out", type=Path, default=REPO / "outputs" / "chunk_yolo_input_ab")
    ap.add_argument("--gallery", type=int, default=12,
                    help="Max overlay pairs to write (worst IoU delta first)")
    args = ap.parse_args()

    pairs = iter_split_pairs(args.dataset, args.splits)
    if not pairs:
        print(f"No images found under {args.dataset} splits={args.splits}")
        sys.exit(1)

    cfg = Config()
    cfg.chunk_detector_priority = ["yolo"]  # force YOLO; no classical in A/B
    # logo suppress irrelevant for YOLO-only
    cfg.dev_logo_yolo_suppress = False

    out = args.out
    (out / "overlays").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    overlays: list[tuple[float, str, np.ndarray]] = []  # |iou_delta|, stem, panel

    print(f"Evaluating {len(pairs)} images from {args.splits} …")
    for i, (img_path, lbl_path) in enumerate(pairs):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  skip unreadable {img_path.name}")
            continue
        h, w = img.shape[:2]
        gt_full = yolo_seg_label_to_mask(lbl_path, h, w)

        roi_mask, _ = detect_container(img, cfg, prefer="yolo")
        gt = cv2.bitwise_and(gt_full, roi_mask)

        preds: dict[str, np.ndarray] = {}
        for mode in MODES:
            cfg.chunk_yolo_input = mode
            pred, det = detect_chunk(img, roi_mask, cfg, prefer="yolo")
            if det != "yolo":
                print(f"  WARN {img_path.stem}: expected yolo got {det!r}")
            preds[mode] = cv2.bitwise_and(pred, roi_mask)

        row: dict = {
            "stem": img_path.stem,
            "split": img_path.parent.name,
            "gt_flag": bool(gt.any()),
            "gt_px": int((gt > 0).sum()),
        }
        metrics = {}
        for mode in MODES:
            m = mask_metrics(preds[mode], gt)
            metrics[mode] = m
            for k, v in m.items():
                if isinstance(v, bool):
                    row[f"{mode}_{k}"] = v
                elif isinstance(v, float):
                    row[f"{mode}_{k}"] = round(v, 6)
                else:
                    row[f"{mode}_{k}"] = v
        rows.append(row)

        # gallery: largest |IoU(full) - IoU(roi)| first
        delta = abs(metrics["full_filter"]["iou"] - metrics["roi_crop"]["iou"])
        if len(overlays) < args.gallery or delta > overlays[-1][0]:
            panel = _panel(img, roi_mask, gt, preds["full_filter"], preds["roi_crop"],
                           metrics, img_path.stem)
            overlays.append((delta, img_path.stem, panel))
            overlays.sort(key=lambda t: -t[0])
            overlays = overlays[: args.gallery]

        if (i + 1) % 10 == 0 or i + 1 == len(pairs):
            print(f"  {i + 1}/{len(pairs)}")

    stats = {mode: summarize(rows, mode) for mode in MODES}
    winner = pick_winner(stats)

    # write CSV
    csv_path = out / "scores.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    for delta, stem, panel in overlays:
        cv2.imwrite(str(out / "overlays" / f"{stem}.jpg"), panel)

    # README
    lines = [
        "# Chunk YOLO input A/B",
        "",
        f"Dataset: `{args.dataset.relative_to(REPO)}` splits={args.splits}  n={len(rows)}",
        "",
        f"**Winner: `{winner}`** (mean IoU; image P/R tie-break)",
        "",
        "| mode | mean IoU | median IoU | mean P | mean R | mean F1 | img agree | img TP/FP/FN |",
        "|------|----------|------------|--------|--------|---------|-----------|--------------|",
    ]
    for mode in MODES:
        s = stats[mode]
        mark = " ←" if mode == winner else ""
        lines.append(
            f"| `{mode}`{mark} | {s['mean_iou']:.4f} | {s['median_iou']:.4f} | "
            f"{s['mean_precision']:.4f} | {s['mean_recall']:.4f} | {s['mean_f1']:.4f} | "
            f"{s['img_agree']:.3f} | {s['img_tp']}/{s['img_fp']}/{s['img_fn']} |"
        )
    lines += [
        "",
        "Set `config.chunk_yolo_input` to the winner.",
        f"CSV: `{csv_path.name}`  overlays: `overlays/`",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines))

    print()
    print("\n".join(lines))
    print(f"\nRecommended: chunk_yolo_input = \"{winner}\"")
    print(f"Wrote {out}")


def _panel(
    img: np.ndarray,
    roi_mask: np.ndarray,
    gt: np.ndarray,
    full_pred: np.ndarray,
    roi_pred: np.ndarray,
    metrics: dict,
    stem: str,
) -> np.ndarray:
    def lab(im: np.ndarray, title: str, color=(255, 255, 255)) -> np.ndarray:
        out = im.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(out, title, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return out

    # dim outside ROI on original
    base = img.astype(np.float32)
    base[roi_mask == 0] *= 0.35
    base = base.astype(np.uint8)

    gt_ov = overlay_mask(base, gt, color=(0, 255, 0), alpha=0.5)
    full_ov = overlay_mask(base, full_pred, color=(0, 0, 255), alpha=0.5)
    roi_ov = overlay_mask(base, roi_pred, color=(255, 0, 0), alpha=0.5)
    gap = np.full((img.shape[0], 4, 3), 40, np.uint8)
    mf = metrics["full_filter"]
    mr = metrics["roi_crop"]
    return np.hstack([
        lab(base, stem),
        gap,
        lab(gt_ov, f"GT {int((gt > 0).sum())}px", (0, 255, 0)),
        gap,
        lab(full_ov, f"full IoU {mf['iou']:.3f}", (0, 0, 255)),
        gap,
        lab(roi_ov, f"roi IoU {mr['iou']:.3f}", (255, 0, 0)),
    ])


if __name__ == "__main__":
    main()
