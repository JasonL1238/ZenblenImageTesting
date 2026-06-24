"""Develop a logo/text detector. The 'zenblen' logo is a horizontal row of
similar-height high-contrast strokes; a lone chunk is not. Detect letter-like
MSER regions and keep only those that group into a horizontal text line.

Visualizes on chunk images (should find NO logo) and logo images (should find it).
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CASES = {
    "03c9_CHUNK": "red_pink/UserGrab_03c9c630-0dbe-4ee5-992a-8791fe8694b0_2026_06_14_18_50_22",
    "62ed_CHUNK": "yellow/UserGrab_62ed4ae1-b1c6-4bd2-a4bf-173eb778184f_2026_06_16_16_04_07",
    "3fe5_LOGO": "red_pink/UserGrab_3fe5f4c7-dd16-45e1-b714-cc59ed467d31_2026_06_14_18_09_51",
    "d3e98_LOGO": "red_pink/UserGrab_d3e98918-635a-4697-b2fb-fc07949682d0_2026_06_15_12_03_22",
    "54121_LOGO": "yellow/UserGrab_54121aaf-255d-47a2-82ea-c7f9b028bb52_2026_06_13_20_05_28",
    "db150_LOGO": "red_pink/UserGrab_db150ed5-c550-45d6-b7b5-921c49d7bd22_2026_06_15_12_08_51",
    "2e775_LOGO": "red_pink/UserGrab_2e7754a2-aac6-4c49-910c-fd3741cda10e_2026_06_14_01_48_12",
}


def letter_candidates(gray, roi_mask):
    ys, xs = np.where(roi_mask > 0)
    roi_h = ys.max() - ys.min()
    mser = cv2.MSER_create()
    mser.setMinArea(60)
    mser.setMaxArea(int(0.02 * roi_mask.size))
    regions, _ = mser.detectRegions(gray)
    cands = []
    for r in regions:
        x, y, w, h = cv2.boundingRect(r.reshape(-1, 1, 2))
        cx, cy = x + w / 2, y + h / 2
        if roi_mask[int(cy), int(cx)] == 0:
            continue
        # letter-like geometry
        if not (0.025 * roi_h < h < 0.20 * roi_h):
            continue
        if not (0.12 < w / max(h, 1) < 2.2):
            continue
        cands.append((x, y, w, h, cx, cy))
    return cands, roi_h


def group_text_lines(cands, roi_h, roi_w):
    """Return letter-groups forming a horizontal text line. A word is several
    similar-height marks on a common baseline spanning a wide horizontal extent —
    chunk specks cluster locally and vary in size, so they fail these tests."""
    lines = []
    used = [False] * len(cands)
    cands = sorted(cands, key=lambda c: c[5])  # by cy
    for i, c in enumerate(cands):
        if used[i]:
            continue
        x, y, w, h, cx, cy = c
        group = [c]
        used[i] = True
        for j, d in enumerate(cands):
            if used[j]:
                continue
            if abs(d[5] - cy) < 0.5 * h and 0.55 < d[3] / max(h, 1) < 1.8:
                group.append(d)
                used[j] = True
        if len(group) < 3:
            continue
        gxs = [g[4] for g in group]
        heights = np.array([g[3] for g in group], float)
        span = (max(gxs) - min(gxs)) / roi_w           # horizontal extent of the word
        h_cv = heights.std() / max(heights.mean(), 1)  # letter-height uniformity
        if span >= 0.32 and h_cv < 0.35:
            lines.append(group)
    return lines


def logo_mask(image, roi_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cands, roi_h = letter_candidates(gray, roi_mask)
    xs = np.where(roi_mask > 0)[1]
    roi_w = xs.max() - xs.min()
    lines = group_text_lines(cands, roi_h, roi_w)
    mask = np.zeros(roi_mask.shape, np.uint8)
    for group in lines:
        xs = [g[0] for g in group] + [g[0] + g[2] for g in group]
        ys = [g[1] for g in group] + [g[1] + g[3] for g in group]
        pad = int(0.02 * roi_h)
        cv2.rectangle(mask, (min(xs) - pad, min(ys) - pad), (max(xs) + pad, max(ys) + pad), 255, -1)
    return mask, cands, lines


def main():
    cells = []
    for name, rel in CASES.items():
        stem = rel.split("/")[1]
        img = cv2.imread(f"data/images/{rel}.jpg")
        roi = cv2.imread(f"outputs/roi_cache/{stem}.png", cv2.IMREAD_GRAYSCALE)
        mask, cands, lines = logo_mask(img, roi)
        vis = img.copy()
        for (x, y, w, h, cx, cy) in cands:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 1)
        vis[mask > 0] = (0.5 * vis[mask > 0] + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        cv2.putText(vis, f"{name} lines={len(lines)} cand={len(cands)}", (5, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cells.append(cv2.resize(vis, (240, 320)))
        print(f"{name:12s}: {len(cands)} letter-cands, {len(lines)} text-lines, mask_px={int((mask>0).sum())}")
    rows = [np.hstack(cells[i:i + 4]) for i in range(0, len(cells), 4)]
    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 0, 0, w - r.shape[1], cv2.BORDER_CONSTANT) for r in rows]
    cv2.imwrite("outputs/logo_detect.png", np.vstack(rows))
    print("saved outputs/logo_detect.png")


if __name__ == "__main__":
    main()
