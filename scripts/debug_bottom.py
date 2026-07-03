"""Trace Path 4 (bottom absolute-chroma) + reference-band stats for one image.

Usage: /opt/miniconda3/bin/python scripts/debug_bottom.py <stem> [--roi sam|yolo|both]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoothie_cv.config import Config
from smoothie_cv.roi import crop_to_roi

SAM_CACHE = Path("outputs/roi_cache_sam")
YOLO_CACHE = Path("outputs/roi_cache_yolo")


def trace(image, roi_mask, cfg, tag):
    roi = crop_to_roi(image, roi_mask)
    img, rmask = roi.image, roi.mask
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ys, xs = np.where(rmask > 0)
    y_top, y_bot = int(ys.min()), int(ys.max())
    roi_h = max(y_bot - y_top, 1)

    # Path 4 inputs
    bot_t = max(y_top, y_bot - cfg.dev_bot_n_rows + 1)
    bot_sel = (rmask > 0)[bot_t:y_bot + 1, :]
    bot_px = lab[bot_t:y_bot + 1, :][bot_sel]
    bot_ch = np.sqrt((bot_px[:, 1] - 128.0) ** 2 + (bot_px[:, 2] - 128.0) ** 2)
    ref_t = int(y_top + 0.35 * roi_h); ref_b = int(y_top + 0.55 * roi_h)
    ref_px = lab[ref_t:ref_b + 1, :][(rmask > 0)[ref_t:ref_b + 1, :]]
    ref_L = float(np.median(ref_px[:, 0]))
    ref_ch = float(np.median(np.sqrt((ref_px[:, 1] - 128.0) ** 2 + (ref_px[:, 2] - 128.0) ** 2)))
    bot_med = float(np.median(bot_ch))
    fires = (ref_L >= cfg.dev_bot_min_body_L and ref_ch >= cfg.dev_bot_min_body_chroma
             and bot_med <= cfg.dev_bot_abs_chroma_max)
    print(f"[{tag}] PATH4: body_L={ref_L:.1f} (>={cfg.dev_bot_min_body_L}) "
          f"body_ch={ref_ch:.1f} (>={cfg.dev_bot_min_body_chroma}) "
          f"bot_med_ch={bot_med:.1f} (<={cfg.dev_bot_abs_chroma_max})  fires={fires}")

    # reference-band pass stats: what would it flag?
    mi = (cv2.erode(rmask, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.dev_roi_erode * 2 + 1,) * 2)) > 0)
    ref_top_y = int(y_top + cfg.dev_global_ref_top_frac * roi_h)
    ref_bot_y = int(y_top + cfg.dev_global_ref_bot_frac * roi_h)
    target_top_y = int(y_top + cfg.dev_global_target_top_frac * roi_h)
    ref_sel = mi.copy(); ref_sel[:ref_top_y, :] = False; ref_sel[ref_bot_y:, :] = False
    ref_mean = lab[ref_sel].mean(axis=0)
    dE_ref = np.sqrt(((lab - ref_mean) ** 2).sum(axis=2))
    chroma_px = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)
    ref_chroma = float(np.sqrt((ref_mean[1] - 128.0) ** 2 + (ref_mean[2] - 128.0) ** 2))
    neutral = (ref_chroma - chroma_px) >= cfg.dev_global_chroma_drop
    # hue branch (mirror of _deviation_mask): saturated but hue-shifted vs reference
    ref_hue = np.degrees(np.arctan2(ref_mean[2] - 128.0, ref_mean[1] - 128.0))
    hue_px = np.degrees(np.arctan2(lab[:, :, 2] - 128.0, lab[:, :, 1] - 128.0))
    hue_diff = np.abs(hue_px - ref_hue)
    hue_diff = np.minimum(hue_diff, 360.0 - hue_diff)
    hue_shift = (hue_diff >= cfg.dev_global_hue_deg) & (chroma_px >= cfg.dev_global_hue_min_chroma)
    target = mi.copy(); target[:target_top_y, :] = False
    base = target & (dE_ref >= cfg.dev_global_thr)
    hits = base & (neutral | hue_shift)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        hits.astype(np.uint8) * 255, connectivity=8)
    big = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)
           if stats[i, cv2.CC_STAT_AREA] >= cfg.dev_global_min_area]
    print(f"[{tag}] REFBAND: ref_chroma={ref_chroma:.1f} ref_hue={float(ref_hue):.1f} "
          f"target_px={int(target.sum())} "
          f"hit_px={int(hits.sum())} (neutral={int((base & neutral).sum())} "
          f"hue={int((base & hue_shift).sum())}) "
          f"big_comps(>= {cfg.dev_global_min_area}px)={big}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stem"); ap.add_argument("--roi", default="yolo")
    args = ap.parse_args()
    match = [p for p in sorted(Path("data/images").rglob("*.jpg")) if args.stem in p.stem]
    p = match[0]; img = cv2.imread(str(p)); cfg = Config()
    print(f"image: {p.stem[:40]}")
    for tag, cache in [("sam", SAM_CACHE), ("yolo", YOLO_CACHE)]:
        if args.roi not in (tag, "both"):
            continue
        rp = cache / f"{p.stem}.png"
        if rp.exists():
            trace(img, cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE), cfg, tag)


if __name__ == "__main__":
    main()
