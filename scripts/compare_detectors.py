#!/usr/bin/env python3
"""
Compare container-detection METHODS on every image, side by side per smoothie.

Methods:
  * classical  — colour-threshold detector (smoothie_cv.detection.classical)
  * sam        — SAM2 fixed-centre-prompt detector (smoothie_cv.detection.sam)
  * sam_flat   — SAM2 body + straight-line top prior (flatten_roi_top)

SAM runs ONCE per image; sam_flat is derived by flattening that mask.

Output — one subfolder per smoothie, all methods together:

    outputs/detect_compare_<timestamp>/
        <shade>/<smoothie_stem>/
            <stem>_classical_roi.png
            <stem>_sam_roi.png
            <stem>_sam_flat_roi.png
            <stem>_sidebyside.png      (classical | sam | sam_flat, labelled)
            <stem>_metrics.json
        gallery.html                   (every smoothie, methods side by side)

Usage:
    python scripts/compare_detectors.py              # all images
    python scripts/compare_detectors.py --sample     # 8-image stratified set
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smoothie_cv.config import Config
from smoothie_cv.detection import (
    draw_container_overlay, flatten_roi_top, top_edge_roughness, _classify_smoothie,
)
from smoothie_cv.detection.classical import detect_classical
from smoothie_cv.detection.sam import detect_sam

ROOT = Path(__file__).resolve().parents[1]

METHODS = ["classical", "sam", "sam_flat"]
_TAG_COLOR = {"classical": (60, 80, 200), "sam": (235, 111, 31), "sam_flat": (67, 160, 46)}

# stratified sample for --sample (tan-jagged / vivid / red)
SAMPLE_IDS = ["054671e6", "62ed4ae1", "6a9960fb", "5142385b",
              "f0b6a6d1", "00891aba", "03c9c630", "09291f76"]


def detect_all(image: np.ndarray, config: Config) -> dict:
    """Run every method once and return {method: (mask, bbox, meta)}."""
    out: dict = {}
    fa = float(image.shape[0] * image.shape[1])

    t = time.perf_counter()
    cmask, cbbox = detect_classical(image)
    out["classical"] = (cmask, cbbox, (time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    smask, sbbox = detect_sam(image, config, flatten_top=False)
    sam_ms = (time.perf_counter() - t) * 1000
    out["sam"] = (smask, sbbox, sam_ms)

    fmask, fbbox = flatten_roi_top(smask)         # derive — no second SAM call
    out["sam_flat"] = (fmask, fbbox, sam_ms)

    result = {}
    for m, (mask, bbox, ms) in out.items():
        result[m] = {
            "mask": mask, "bbox": bbox,
            "ms": round(ms, 1),
            "area_frac": round(float((mask > 0).sum()) / fa, 3),
            "roughness": round(top_edge_roughness(mask), 2),
        }
    return result


def labelled(overlay: np.ndarray, text: str, color, height: int = 360) -> np.ndarray:
    """Resize an overlay to a common height and stamp a label bar on top."""
    h, w = overlay.shape[:2]
    vis = cv2.resize(overlay, (int(w * height / h), height), interpolation=cv2.INTER_AREA)
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 22), color, -1)
    cv2.putText(vis, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def side_by_side(image: np.ndarray, results: dict) -> np.ndarray:
    """Stitch each method's overlay horizontally with labels + separators."""
    panels = []
    for m in METHODS:
        ov = draw_container_overlay(image, results[m]["mask"])
        tag = f"{m}  rough={results[m]['roughness']:.1f}"
        panels.append(labelled(ov, tag, _TAG_COLOR[m]))
        panels.append(np.full((panels[0].shape[0], 3, 3), 255, np.uint8))
    return cv2.hconcat(panels[:-1])


def uri(img_bgr: np.ndarray, width: int = 720) -> str:
    h, w = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, (width, int(h * width / w)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def build_gallery(out_dir: Path, rows: list[dict]) -> Path:
    cards = []
    for r in rows:
        stats = " · ".join(
            f'<b>{m}</b> {r["metrics"][m]["roughness"]:.1f}px/{r["metrics"][m]["area_frac"]:.2f}'
            for m in METHODS
        )
        cards.append(
            f'<figure class="card"><figcaption>{r["id"]} '
            f'<span class="shade">{r["shade"]}</span></figcaption>'
            f'<img src="{uri(r["composite"])}">'
            f'<div class="stats">{stats}</div></figure>'
        )
    html = f"""<title>Detection methods per smoothie — classical | sam | sam_flat</title>
<style>
 body{{margin:0;font:13px/1.45 -apple-system,system-ui,sans-serif;background:#0e1116;color:#e6edf3}}
 header{{padding:20px 26px;border-bottom:1px solid #222;background:#11161d}}
 h1{{margin:0 0 6px;font-size:18px}} .meta{{color:#9aa7b4}}
 .legend{{margin-top:8px;font-size:12px}}
 .chip{{display:inline-block;padding:2px 8px;border-radius:3px;color:#fff;margin-right:8px;font-weight:600}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:16px;padding:18px 26px}}
 .card{{background:#161b22;border:1px solid #222;border-radius:8px;overflow:hidden;margin:0}}
 figcaption{{padding:7px 10px;font-family:ui-monospace,monospace;font-size:12px;border-bottom:1px solid #222}}
 .shade{{color:#7d8896;margin-left:6px}}
 .card img{{display:block;width:100%;height:auto}}
 .stats{{padding:6px 10px;font-size:11px;color:#9aa7b4;font-family:ui-monospace,monospace}}
 .stats b{{color:#cdd9e5}}
</style>
<header><h1>Container detection per smoothie — classical | sam | sam_flat</h1>
<div class="meta">Green contour = detected ROI · {len(rows)} smoothies · each row stitches the three methods left→right · stats = top-edge roughness(px)/area-fraction</div>
<div class="legend">
 <span class="chip" style="background:rgb(200,80,60)">classical</span>
 <span class="chip" style="background:rgb(31,111,235)">sam</span>
 <span class="chip" style="background:rgb(46,160,67)">sam_flat</span></div>
</header>
<div class="grid">{''.join(cards)}</div>
"""
    p = out_dir / "gallery.html"
    p.write_text(html)
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="use 8-image stratified set")
    args = ap.parse_args()

    config = Config()
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = ROOT / "outputs" / f"detect_compare_{stamp}"

    if args.sample:
        paths = []
        for sid in SAMPLE_IDS:
            hits = list((ROOT / "data/images").rglob(f"*{sid}*.jpg"))
            if hits:
                paths.append(hits[0])
    else:
        paths = sorted((ROOT / "data/images").rglob("*.jpg"))

    print(f"Comparing {METHODS} on {len(paths)} images → {out_dir}\n")

    rows = []
    for i, p in enumerate(paths, 1):
        image = cv2.imread(str(p))
        if image is None:
            continue
        shade = _classify_smoothie(image).value
        sid = p.stem.split("_")[1][:8]
        results = detect_all(image, config)

        smoothie_dir = out_dir / shade / p.stem
        smoothie_dir.mkdir(parents=True, exist_ok=True)
        for m in METHODS:
            cv2.imwrite(str(smoothie_dir / f"{p.stem}_{m}_roi.png"),
                        draw_container_overlay(image, results[m]["mask"]))
        composite = side_by_side(image, results)
        cv2.imwrite(str(smoothie_dir / f"{p.stem}_sidebyside.png"), composite)

        metrics = {m: {k: results[m][k] for k in ("ms", "area_frac", "roughness")}
                   for m in METHODS}
        (smoothie_dir / f"{p.stem}_metrics.json").write_text(
            json.dumps({"image": str(p), "shade": shade, "methods": metrics}, indent=2))

        rows.append({"id": sid, "shade": shade, "composite": composite, "metrics": metrics})
        print(f"  [{i:>3}/{len(paths)}] {sid} ({shade:12}) "
              f"rough classical={metrics['classical']['roughness']:.1f} "
              f"sam={metrics['sam']['roughness']:.1f} sam_flat={metrics['sam_flat']['roughness']:.1f}")

    gallery = build_gallery(out_dir, rows)
    print(f"\nWrote {len(rows)} per-smoothie folders under {out_dir}")
    print(f"Gallery → {gallery}")


if __name__ == "__main__":
    main()
