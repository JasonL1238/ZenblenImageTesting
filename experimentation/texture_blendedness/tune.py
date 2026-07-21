"""
Tuning harness: run YOLO ONCE on a small curated set, then evaluate many
scoring variants on the cached deviation maps. Goal: make clearly-unblended
frames (e.g. 227004) score < 50 while genuinely-smooth ones stay high.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from texture import TextureParams, deviation_map, score_from_map  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
IMG = REPO / "training" / "labeling" / "data" / "images"
STD = REPO / "training" / "checkpoints" / "yolo_standard_seg.pt"
LOGO = REPO / "training" / "checkpoints" / "yolo_logo_seg.pt"

# curated set with expected verdicts
CASES = [
    ("227004", "BAD  clumpy grey  -> want <50"),
    ("226583", "BAD  streaky      -> want low"),
    ("227068", "BAD  green streaky -> want low"),
    ("226872", "BAD  streaky      -> want low"),
    ("226920", "GOOD smooth pink  -> want high"),
    ("226743", "GOOD (logo heavy) -> want high"),
    ("226383", "GOOD              -> want high"),
]


def _models():
    from ultralytics import YOLO
    return YOLO(str(STD)), YOLO(str(LOGO))


def _roi(m, image):
    h, w = image.shape[:2]
    r = m(image, verbose=False, device="cpu")[0]
    if r.masks is None or len(r.masks) == 0:
        return np.zeros((h, w), np.uint8)
    raw = r.masks.data[int(np.argmax(r.boxes.conf.cpu().numpy()))].cpu().numpy()
    mm = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
    ff = ((mm > 0.5) * 255).astype(np.uint8)
    fl = ff.copy()
    cv2.floodFill(fl, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return ff | cv2.bitwise_not(fl)


def _logo(m, image):
    h, w = image.shape[:2]
    r = m(image, verbose=False, device="cpu", conf=0.25)[0]
    mask = np.zeros((h, w), np.uint8)
    if r.masks is None:
        return mask
    for raw in r.masks.data.cpu().numpy():
        mm = cv2.resize(raw, (w, h), interpolation=cv2.INTER_NEAREST)
        mask[mm > 0.5] = 255
    return mask


def main():
    std, logo = _models()
    # base params control the MAP (blur/thresh/open); scoring params are swept
    base = TextureParams(blur_kernel=121, delta_e_thresh=6.0, open_kernel_px=7)

    # precompute deviation maps once per image
    cache = {}
    for stem, _ in CASES:
        p = IMG / f"{stem}.jpg"
        img = cv2.imread(str(p))
        roi, lg = _roi(std, img), _logo(logo, img)
        dm, reg = deviation_map(img, roi, lg, base)
        frac = float((dm > 0).sum()) / max(1, int(reg.sum()))
        mean = float(dm[reg].mean()) if int(reg.sum()) else 0.0
        cache[stem] = (dm, reg, frac, mean)

    print("\nRaw components (map: thresh=6, open=7):")
    print(f"{'img':8} {'flagged%':>9} {'meanΔE':>7}  note")
    for stem, note in CASES:
        _, _, frac, mean = cache[stem]
        print(f"{stem:8} {frac*100:8.2f}% {mean:7.2f}  {note}")

    variants = [
        ("area k4", dict(agg="area", strictness=4.0)),
        ("area k6", dict(agg="area", strictness=6.0)),
        ("area k8", dict(agg="area", strictness=8.0)),
        ("sev  k4", dict(agg="sev", strictness=4.0)),
        ("sev  k6", dict(agg="sev", strictness=6.0)),
    ]
    print("\nScores by variant:")
    hdr = f"{'img':8} " + " ".join(f"{n:>8}" for n, _ in variants)
    print(hdr)
    for stem, note in CASES:
        dm, reg, _, _ = cache[stem]
        cells = []
        for _, kw in variants:
            s = score_from_map(dm, reg, replace(base, **kw))
            cells.append(f"{s:8.1f}")
        print(f"{stem:8} " + " ".join(cells) + f"  {note}")


if __name__ == "__main__":
    main()
