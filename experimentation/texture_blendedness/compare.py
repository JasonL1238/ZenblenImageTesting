"""
Comparison harness. Runs EVERY method in methods/ on the same sampled images
(YOLO masks computed once and cached) and writes:

  outputs/comparison/scores_all.csv   one row per image, one column per method
  outputs/comparison/panels/<...>.jpg original | method_A | method_B | ...

Sort order: by the reference method (--sort, default dev_area) ascending, so the
least-blended images (per that method) are on top.

Run:
  /opt/miniconda3/bin/python experimentation/texture_blendedness/compare.py --n 100
"""
from __future__ import annotations

import argparse
import csv
import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import methods as methods_pkg  # noqa: E402

OUT = HERE / "outputs" / "comparison"


def load_methods(only=None):
    mods = []
    for m in pkgutil.iter_modules(methods_pkg.__path__):
        mod = importlib.import_module(f"methods.{m.name}")
        if not (hasattr(mod, "score") and hasattr(mod, "NAME")):
            continue
        if only and mod.NAME not in only:
            continue
        mods.append(mod)
    mods.sort(key=lambda x: x.NAME)
    return mods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--sort", default="dev_area", help="method NAME to rank by")
    ap.add_argument("--only", nargs="*", help="restrict to these method NAMEs")
    args = ap.parse_args()

    mods = load_methods(args.only)
    if not mods:
        print("No methods found in methods/")
        return
    names = [m.NAME for m in mods]
    print(f"Methods: {names}")

    images = common.sample_images(args.n)
    (OUT / "panels").mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(images, 1):
        image, roi, logo = common.get_masks(path)
        if int((roi > 0).sum()) == 0:
            print(f"  [{i}/{len(images)}] no_roi {path.name}")
            continue
        panels = [image.copy()]
        cv2.rectangle(panels[0], (0, 0), (image.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(panels[0], "original", (8, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        row = {"image": path.stem, "source": str(path.relative_to(common.REPO))}
        for mod in mods:
            try:
                s, flag01 = mod.score(image, roi, logo)
            except Exception:
                traceback.print_exc()
                s, flag01 = float("nan"), np.zeros(image.shape[:2], np.float32)
            row[mod.NAME] = round(float(s), 2)
            panels.append(common.render_panel(image, flag01, roi, s, mod.NAME))
        rows.append(row)
        cv2.imwrite(str(OUT / "panels" / f"{path.stem}.jpg"), np.hstack(panels))
        print(f"  [{i}/{len(images)}] " +
              "  ".join(f"{n}={row[n]:.0f}" for n in names) + f"  {path.name}")

    # rank by the reference method (missing -> bottom)
    key = args.sort if args.sort in names else names[0]
    rows.sort(key=lambda r: r.get(key, 1e9))
    # rename panels with a rank prefix
    for rank, r in enumerate(rows, 1):
        src = OUT / "panels" / f"{r['image']}.jpg"
        dst = OUT / "panels" / f"{rank:03d}_{key}{r.get(key, 0):05.1f}_{r['image']}.jpg"
        if src.exists():
            src.rename(dst)
        r["panel"] = dst.name
        r["rank"] = rank

    fields = ["rank"] + names + ["image", "panel", "source"]
    with open(OUT / "scores_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\nDone. {len(rows)} scored. Compare: {OUT/'scores_all.csv'}  "
          f"and {OUT/'panels'}/")
    for n in names:
        vals = np.array([r[n] for r in rows if isinstance(r.get(n), (int, float))
                         and r[n] == r[n]])
        if len(vals):
            print(f"  {n:14} min={vals.min():5.1f} med={np.median(vals):5.1f} "
                  f"max={vals.max():5.1f}")


if __name__ == "__main__":
    main()
