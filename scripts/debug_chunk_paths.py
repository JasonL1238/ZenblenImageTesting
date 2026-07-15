"""Instrumented single-image chunk-detector trace.

Re-runs the deviation detector's component loop with full per-component logging:
which components exist after thresholding/exclusions, which path (compact/dark/
chroma) accepted each, and what the logo text-line detector saw. Uses the
cached YOLO ROI by default.

Usage:
  /opt/miniconda3/bin/python scripts/debug_chunk_paths.py <stem-prefix>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smoothie_cv.config import Config
from smoothie_cv.pipelines.classical_cv import ClassicalCVPipeline
from smoothie_cv.roi import crop_to_roi

YOLO_CACHE = Path("outputs/roi_cache_yolo")


def trace(image: np.ndarray, roi_mask: np.ndarray, cfg: Config, tag: str,
          out_dir: Path) -> None:
    pipe = ClassicalCVPipeline(cfg)
    roi = crop_to_roi(image, roi_mask)
    img, rmask = roi.image, roi.mask

    # trained-logo mask (mirror of _deviation_mask): full-frame logo → crop to ROI bbox
    logo_crop = None
    if cfg.dev_logo_yolo_suppress:
        from smoothie_cv.detection.logo import detect_logo
        full_logo = detect_logo(image, cfg)
        x0, y0 = roi.offset
        ch, cw = roi.mask.shape
        logo_crop = full_logo[y0:y0 + ch, x0:x0 + cw]

    m = (rmask > 0).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ek = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (cfg.dev_roi_erode * 2 + 1, cfg.dev_roi_erode * 2 + 1))
    interior = cv2.erode(rmask, ek, iterations=1)
    mi = interior > 0

    ys, xs = np.where(rmask > 0)
    y_top, y_bot = int(ys.min()), int(ys.max())
    x_left, x_right = int(xs.min()), int(xs.max())
    roi_h = max(y_bot - y_top, 1)
    roi_w = max(x_right - x_left, 1)
    foam_cut = int(y_top + cfg.dev_foam_frac * roi_h)
    exempt_y = (int(y_top + (1.0 - cfg.dev_bright_bot_exempt_frac) * roi_h)
                if cfg.dev_bright_bot_exempt_frac > 0 else y_bot + 1)

    L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)

    # two-pass base (mirror of ClassicalCVPipeline._deviation_mask)
    K = cfg.dev_blur_kernel | 1

    def _masked_base(weight):
        den = cv2.GaussianBlur(weight, (K, K), 0) + 1e-6
        return np.dstack([cv2.GaussianBlur(lab[:, :, i] * weight, (K, K), 0) / den
                          for i in range(3)])

    base = _masked_base(m)
    print_px = (L - base[:, :, 0] > cfg.dev_bright_dL) & (chroma < cfg.dev_bright_chroma)
    print_px[exempt_y:, :] = False
    if print_px.any():
        base = _masked_base(m * (~print_px))

    dE = np.sqrt(((lab - base) ** 2).sum(axis=2))
    vals = dE[mi]
    thr = max(float(vals.mean() + cfg.dev_k_sigma * vals.std()), cfg.dev_min_delta_e)
    raw = ((dE >= thr) & mi).astype(np.uint8) * 255

    base_chroma = np.sqrt((base[:, :, 1] - 128.0) ** 2 + (base[:, :, 2] - 128.0) ** 2)
    dC = chroma - base_chroma
    glare = (L > cfg.dev_glare_L) & (chroma < cfg.dev_glare_chroma)
    raw[glare] = 0
    dL = L - base[:, :, 0]
    bright_neutral = (dL > cfg.dev_bright_dL) & (chroma < cfg.dev_bright_chroma)
    bright_neutral[exempt_y:, :] = False
    raw[bright_neutral] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    clean[:foam_cut, :] = 0

    d = cfg.dev_dark_print_adj_dilate
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1))
    print_halo = cv2.dilate(
        cv2.morphologyEx(bright_neutral.astype(np.uint8), cv2.MORPH_CLOSE, ck), ck) > 0

    n, labels, stats, cents = cv2.connectedComponentsWithStats(clean, connectivity=8)
    logo_labels = pipe._logo_text_labels(stats, cents, n, roi_h, roi_w)

    # confirmed-wordmark band (mirror of _deviation_mask): letter-sized components
    # whose centroid lands inside a detected wordmark's band are print footprint.
    logo_band = None
    if cfg.dev_logo_band_suppress and logo_labels:
        ll = list(logo_labels)
        my = cfg.dev_logo_band_margin_frac * float(
            np.median([stats[i, cv2.CC_STAT_HEIGHT] for i in ll]))
        med_letter_area = float(np.median([stats[i, cv2.CC_STAT_AREA] for i in ll]))
        logo_band = (min(stats[i, cv2.CC_STAT_TOP] for i in ll) - my,
                     max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
                         for i in ll) + my,
                     min(stats[i, cv2.CC_STAT_LEFT] for i in ll),
                     max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
                         for i in ll),
                     cfg.dev_logo_band_max_area_mult * med_letter_area)

    print(f"\n===== {tag} =====")
    print(f"ROI bbox: x[{x_left},{x_right}] y[{y_top},{y_bot}]  h={roi_h} w={roi_w}")
    print(f"thr={thr:.2f} (mean={vals.mean():.2f} std={vals.std():.2f})  "
          f"foam_cut_y={foam_cut}  components={n - 1}  logo_labels={sorted(logo_labels)}")

    # logo candidate trace
    print("-- letter candidates (area>=%d, h in [%.3f,%.3f]*roi_h, 0.12<w/h<2.4):"
          % (cfg.dev_letter_min_area, cfg.dev_letter_h_lo, cfg.dev_letter_h_hi))
    for i in range(1, n):
        h = int(stats[i, cv2.CC_STAT_HEIGHT]); w = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])
        ok_a = area >= cfg.dev_letter_min_area
        ok_h = cfg.dev_letter_h_lo * roi_h < h < cfg.dev_letter_h_hi * roi_h
        ok_ar = 0.12 < w / max(h, 1) < 2.4
        if ok_a and (h > 8):  # only print non-trivial comps
            print(f"   comp {i:3d}: area={area:5d} w={w:3d} h={h:3d} "
                  f"cx={cents[i][0]:6.1f} cy={cents[i][1]:6.1f} "
                  f"cand={'Y' if (ok_a and ok_h and ok_ar) else 'n'}"
                  f" (a={ok_a} h={ok_h} ar={ok_ar})")

    roi_area = int((rmask > 0).sum())
    print("-- component path decisions:")
    accepted = []
    for li in range(1, n):
        area = int(stats[li, cv2.CC_STAT_AREA])
        if area < cfg.dev_relaxed_min_area or area > roi_area * cfg.dev_max_area_frac:
            continue
        bw = int(stats[li, cv2.CC_STAT_WIDTH]); bh = int(stats[li, cv2.CC_STAT_HEIGHT])
        comp_mask = labels == li
        comp = comp_mask.astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        cnt = max(cnts, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / hull if hull > 0 else 0.0
        extent = area / float(bw * bh) if bw * bh > 0 else 0.0
        aspect = bw / float(bh) if bh > 0 else 0.0
        is_logo = li in logo_labels
        if (logo_band is not None and area <= logo_band[4]
                and logo_band[0] <= cents[li][1] <= logo_band[1]
                and logo_band[2] <= cents[li][0] <= logo_band[3]):
            print(f"   comp {li:3d}: area={area:5d} cy={cents[li][1]:6.1f} "
                  f"cx={cents[li][0]:6.1f}  rej:logo_band")
            continue
        if logo_crop is not None:
            inside = int((comp_mask & (logo_crop > 0)).sum())
            if inside / max(area, 1) >= cfg.dev_logo_yolo_overlap:
                print(f"   comp {li:3d}: area={area:5d} cy={cents[li][1]:6.1f} "
                      f"cx={cents[li][0]:6.1f}  rej:logo_yolo "
                      f"(overlap={inside/max(area,1):.2f})")
                continue
        if aspect < cfg.dev_aspect_lo:
            verdict = "rej:aspect_lo"
        else:
            mdL = float(dL[comp_mask].mean()); mdC = float(dC[comp_mask].mean())
            mdE = float(dE[comp_mask].mean())
            if (mdL > cfg.dev_bright_dL and mdC < cfg.dev_comp_bright_dC_max
                    and cents[li][1] < exempt_y):
                print(f"   comp {li:3d}: area={area:5d} dL={mdL:+6.1f} dC={mdC:+6.1f} "
                      f"cy={cents[li][1]:6.1f}  rej:bright_desat")
                continue
            halo_frac = float(print_halo[comp_mask].mean())
            compact = (solidity >= cfg.dev_min_solidity and extent >= cfg.dev_min_extent
                       and aspect <= cfg.dev_aspect_hi and area >= cfg.dev_min_area
                       and mdE >= cfg.dev_compact_min_delta_e)
            dark = (mdL <= cfg.dev_dark_dL and solidity >= cfg.dev_dark_min_solidity
                    and extent >= cfg.dev_dark_min_extent
                    and halo_frac < cfg.dev_dark_print_adj_frac
                    and cents[li][1] >= y_top + cfg.dev_relaxed_top_frac * roi_h)
            chromatic = (mdC >= cfg.dev_chroma_dC
                         and mdL <= cfg.dev_chroma_dL_max
                         and solidity >= cfg.dev_dark_min_solidity
                         and extent >= cfg.dev_dark_min_extent)
            colour_cued = (dark or chromatic) and aspect <= cfg.dev_relaxed_aspect_hi
            paths = [p for p, v in
                     [("compact", compact), ("dark", dark and colour_cued),
                      ("chroma", chromatic and colour_cued)] if v]
            verdict = "ACCEPT:" + "+".join(paths) if paths else "rej:no_path"
            print(f"   comp {li:3d}: area={area:5d} sol={solidity:.2f} ext={extent:.2f} "
                  f"asp={aspect:.2f} dL={mdL:+6.1f} dC={mdC:+6.1f} dE={mdE:5.1f} "
                  f"halo={halo_frac:.2f} cy={cents[li][1]:6.1f} "
                  f"logo={'Y' if is_logo else 'n'}  {verdict}")
            if paths and not is_logo:
                accepted.append(li)
    print(f"accepted (non-logo): {accepted}")

    # save debug images
    out_dir.mkdir(parents=True, exist_ok=True)
    dE_vis = np.clip(dE / 40.0 * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / f"{tag}_dE.png"), cv2.applyColorMap(dE_vis, cv2.COLORMAP_INFERNO))
    cv2.imwrite(str(out_dir / f"{tag}_clean.png"), clean)
    full = pipe.analyze  # not used; keep parity


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem")
    ap.add_argument("--roi-cache", default=str(YOLO_CACHE),
                    help="Directory of cached YOLO ROI masks (default: outputs/roi_cache_yolo)")
    args = ap.parse_args()

    imgs = sorted(Path("data/images").rglob("*.jpg"))
    match = [p for p in imgs if args.stem in p.stem]
    if not match:
        print("no image match"); sys.exit(1)
    p = match[0]
    print(f"image: {p}")
    img = cv2.imread(str(p))
    cfg = Config()
    out_dir = Path("outputs/debug_chunks") / p.stem[:17]

    cache = Path(args.roi_cache)
    rp = cache / f"{p.stem}.png"
    if not rp.exists():
        print(f"(ROI not cached at {rp}, skip)"); sys.exit(1)
    roi_mask = cv2.imread(str(rp), cv2.IMREAD_GRAYSCALE)
    trace(img, roi_mask, cfg, "yolo", out_dir)


if __name__ == "__main__":
    main()
