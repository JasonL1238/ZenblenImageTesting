"""
Classical CV pipeline for smoothie blendedness detection.

Primary detector is colour-agnostic **local-deviation** segmentation: a chunk is
a contiguous patch whose LAB colour differs from the smoothie's *local* base
colour. This keys off relative deviation, not any absolute colour, so it
generalises across red/pink/yellow/etc. without per-shade tuning. A well-blended
smoothie is locally uniform → deviation ≈ 0 everywhere → score ≈ 1.0.

Canny edge detection remains available (`classical_method="canny"`) as a
boundary-oriented alternative, but it only catches chunk *rims*, not bodies.
"""

from __future__ import annotations

import cv2
import numpy as np

from smoothie_cv.config import Config
from smoothie_cv.pipelines.base import BlendPipeline, BlendResult
from smoothie_cv.roi import crop_to_roi, paste_mask
from smoothie_cv.scoring.metrics import compute_blend_score


class ClassicalCVPipeline(BlendPipeline):

    name = "classical"

    def __init__(self, config: Config) -> None:
        self.config = config

    def analyze(self, image: np.ndarray, roi_mask: np.ndarray | None = None) -> BlendResult:
        h, w = image.shape[:2]
        if roi_mask is None:
            roi_mask = np.full((h, w), 255, dtype=np.uint8)

        roi = crop_to_roi(image, roi_mask)

        if self.config.classical_method == "canny":
            unblended = self._edge_mask(roi.image, roi.mask)
        else:
            unblended = self._deviation_mask(roi.image, roi.mask)

        # clip to ROI and map back to full-frame coords
        unblended = cv2.bitwise_and(unblended, roi.mask)
        unblended = paste_mask(unblended, roi)
        score = compute_blend_score(unblended, roi_mask)
        passed = score >= self.config.threshold

        return BlendResult(
            blend_score=score,
            passed=passed,
            mask=unblended,
            pipeline_name=self.name,
            metadata={
                "method": self.config.classical_method,
                "dev_k_sigma": self.config.dev_k_sigma,
            },
        )

    # ── primary: colour-agnostic local-deviation detector ──────────────────────
    def _deviation_mask(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Flag contiguous patches whose LAB colour deviates from the local base.

        Steps:
          1. LAB convert (perceptual — distance ≈ "how different it looks").
          2. Local base colour = masked large-kernel blur (smooth lighting/colour
             gradient of a well-blended smoothie). The masked blur normalises by
             the blurred mask so black pixels outside the contour don't bleed in
             and fake a deviation ring at the ROI boundary.
          3. ΔE = per-pixel distance to that base → anomaly map.
          4. Adaptive threshold (mean + k·σ of ΔE inside the ROI) — relative to
             each image, so no per-colour constant.
          5. Drop glare (bright, low-chroma specular) and morph-clean.
          6. Keep blobs by area + solidity (chunks are compact; logo strokes and
             texture speckle are thin / small).
        """
        cfg = self.config
        m = (roi_mask > 0).astype(np.float32)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

        # erode the ROI a little so the hard contour edge itself isn't measured
        erode_px = cfg.dev_roi_erode
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        interior = cv2.erode(roi_mask, ek, iterations=1)
        mi = (interior > 0)

        # masked large blur → local base colour (avoids outside-contour black bleed)
        K = cfg.dev_blur_kernel | 1  # force odd
        denom = cv2.GaussianBlur(m, (K, K), 0) + 1e-6
        base = np.dstack([cv2.GaussianBlur(lab[:, :, i] * m, (K, K), 0) / denom for i in range(3)])
        dE = np.sqrt(((lab - base) ** 2).sum(axis=2))

        vals = dE[mi]
        if vals.size == 0:
            return np.zeros(roi_mask.shape, np.uint8)

        thr = float(vals.mean() + cfg.dev_k_sigma * vals.std())
        thr = max(thr, cfg.dev_min_delta_e)  # floor: ignore trivially small deviations
        raw = ((dE >= thr) & mi).astype(np.uint8) * 255

        # drop specular glare: very bright + low chroma (LAB a,b centred at 128)
        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)
        base_chroma = np.sqrt((base[:, :, 1] - 128.0) ** 2 + (base[:, :, 2] - 128.0) ** 2)
        dC = chroma - base_chroma  # +ve = more saturated than local base (coloured chunk)
        glare = (L > cfg.dev_glare_L) & (chroma < cfg.dev_glare_chroma)
        raw[glare] = 0

        # drop the printed-logo / backlit-text signature: regions much BRIGHTER
        # than the local base with low chroma (cream/neutral text). Real chunks are
        # darker than base or strongly coloured, so this spares them while killing
        # the dominant "zenblen" logo false positive on dark smoothies.
        dL = L - base[:, :, 0]
        bright_neutral = (dL > cfg.dev_bright_dL) & (chroma < cfg.dev_bright_chroma)
        raw[bright_neutral] = 0

        # despeckle (small open only). No close — keep logo letters as separate
        # components so the text-line detector can recognise their arrangement.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)

        # exclude the top foam/rim band of the cup (bubbles read as deviation)
        ys, xs = np.where(roi_mask > 0)
        y_top, y_bot = ys.min(), ys.max()
        x_left, x_right = xs.min(), xs.max()
        roi_h = max(y_bot - y_top, 1)
        roi_w = max(x_right - x_left, 1)
        foam_cut = int(y_top + cfg.dev_foam_frac * roi_h)
        clean[:foam_cut, :] = 0

        # connected components → classify the "zenblen" logo (a horizontal row of
        # similar-height marks spanning a wide extent) and exclude it, then keep
        # the remaining compact, solid blobs as chunks.
        n, labels, stats, cents = cv2.connectedComponentsWithStats(clean, connectivity=8)
        logo_labels = self._logo_text_labels(stats, cents, n, roi_h, roi_w)

        # per-pixel deviation vector from local base (for directional growth below)
        dev = lab - base

        roi_area = int((roi_mask > 0).sum())
        out = np.zeros(roi_mask.shape, np.uint8)
        grow_seed_masks: list[np.ndarray] = []  # seeds large enough to grow
        for lab_i in range(1, n):
            if lab_i in logo_labels:
                continue
            area = int(stats[lab_i, cv2.CC_STAT_AREA])
            # area ceiling is common; the floor differs by path (colour-cued paths
            # accept smaller flecks), so only reject on the ceiling / relaxed floor here.
            if area < cfg.dev_relaxed_min_area or area > roi_area * cfg.dev_max_area_frac:
                continue
            bw = int(stats[lab_i, cv2.CC_STAT_WIDTH])
            bh = int(stats[lab_i, cv2.CC_STAT_HEIGHT])
            comp_mask = labels == lab_i
            comp = comp_mask.astype(np.uint8)
            cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            hull = cv2.contourArea(cv2.convexHull(cnt))
            solidity = area / hull if hull > 0 else 0.0
            extent = area / float(bw * bh) if bw * bh > 0 else 0.0
            aspect = bw / float(bh) if bh > 0 else 0.0

            if aspect < cfg.dev_aspect_lo:
                continue
            mean_dL = float(dL[comp_mask].mean())
            mean_dC = float(dC[comp_mask].mean())
            mean_dE = float(dE[comp_mask].mean())

            # Path 1 — compact solid blob (clear chunks, any colour). Strict shape +
            # strict aspect + strict area, and the original ΔE floor: this path carries
            # no colour cue, so its precision rests on shape + a strong deviation. (The
            # lower global floor exists only to let the colour-cued paths see faint
            # hue-similar chunks; a faint compact blob is more likely glare than a chunk.)
            compact = (solidity >= cfg.dev_min_solidity and extent >= cfg.dev_min_extent
                       and aspect <= cfg.dev_aspect_hi and area >= cfg.dev_min_area
                       and mean_dE >= cfg.dev_compact_min_delta_e)
            # Path 2 — dark coherent deviation: subtle chunks (e.g. a dissolving
            # chunk's dark rim) come back as a thin arc that fails the compact gate,
            # but they are distinctly DARKER than the local base. The cream logo and
            # glare are BRIGHTER, so the darkness gate recovers these without them.
            dark = (mean_dL <= cfg.dev_dark_dL
                    and solidity >= cfg.dev_dark_min_solidity
                    and extent >= cfg.dev_dark_min_extent)
            # Path 3 — chroma deviation: hue-similar chunks (orange-on-yellow, amber
            # flecks) are NOT darker than base so Path 2 misses them, but they are more
            # SATURATED than base. Glare/highlights/logo desaturate (mean_dC < 0), so
            # this gate spares them — same precision principle as the darkness gate.
            chromatic = (mean_dC >= cfg.dev_chroma_dC
                         and solidity >= cfg.dev_dark_min_solidity
                         and extent >= cfg.dev_dark_min_extent)
            # the colour-cued paths (dark/chroma) may be elongated streaks → relaxed
            # aspect; logo strokes are excluded by neutrality (mean_dC<0, mean_dL>0).
            colour_cued = (dark or chromatic) and aspect <= cfg.dev_relaxed_aspect_hi

            if compact or colour_cued:
                out[comp_mask] = 255
                if area >= cfg.dev_grow_min_seed_area:
                    grow_seed_masks.append(comp_mask)

        # directional grow: the gates above mask each chunk's high-contrast CORE; its
        # softer margin/tail (fading toward smoothie colour) deviates from base in the
        # SAME colour direction as the core but with smaller magnitude, so a raw-ΔE
        # threshold drops it and the chunk is only partly covered. Grow each (large) seed
        # into contiguous pixels whose deviation projects strongly onto that seed's own
        # signature direction. Same exclusions (glare/bright-neutral/foam) apply, growth
        # is distance-bounded, and only confident seeds grow — so a marginal speck is
        # never amplified into a flag and growth can't crawl across the cup.
        excluded = glare | bright_neutral
        for seed_mask in grow_seed_masks:
            sig = dev[seed_mask].mean(axis=0)        # chunk's mean deviation direction
            norm = float(np.linalg.norm(sig))
            if norm < 1e-3:
                continue
            proj = (dev * (sig / norm)).sum(axis=2)  # how far each pixel deviates that way
            field = ((proj >= cfg.dev_grow_proj_thr) & mi).astype(np.uint8) * 255
            field[excluded] = 0
            field[:foam_cut, :] = 0
            seed = seed_mask.astype(np.uint8) * 255
            out = cv2.bitwise_or(out, self._reconstruct(seed, field, cfg.dev_grow_max_iter))
        return out

    @staticmethod
    def _reconstruct(seeds: np.ndarray, field: np.ndarray, max_iter: int) -> np.ndarray:
        """Geodesic dilation of `seeds` within `field` (bounded), retaining the seed.
        Recovers a chunk's full extent from its confirmed core."""
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cur = seeds.copy()
        for _ in range(max_iter):
            nxt = cv2.bitwise_and(cv2.dilate(cur, k), field) | seeds
            if np.array_equal(nxt, cur):
                break
            cur = nxt
        return cur

    def _logo_text_labels(self, stats, cents, n, roi_h, roi_w) -> set[int]:
        """Identify component labels that belong to the logo text line.

        The "zenblen" logo is several similar-height marks sharing a baseline and
        spanning a wide horizontal extent. A chunk (or local speck cluster) does
        not: it is one blob, or its parts vary in size and stay local. So we group
        letter-like components by baseline + height similarity and flag a group as
        logo only if it has enough members, spans a wide extent, and has uniform
        letter heights.
        """
        cfg = self.config
        cands = []  # (label, cx, cy, h)
        for i in range(1, n):
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < cfg.dev_letter_min_area:
                continue
            if not (cfg.dev_letter_h_lo * roi_h < h < cfg.dev_letter_h_hi * roi_h):
                continue
            if not (0.12 < w / max(h, 1) < 2.4):
                continue
            cands.append((i, cents[i][0], cents[i][1], h))

        logo: set[int] = set()
        used = [False] * len(cands)
        order = sorted(range(len(cands)), key=lambda j: cands[j][2])  # by cy
        for a in order:
            if used[a]:
                continue
            _, _, cy, h = cands[a]
            group = [a]
            used[a] = True
            for b in order:
                if used[b]:
                    continue
                if abs(cands[b][2] - cy) < 0.5 * h and 0.55 < cands[b][3] / max(h, 1) < 1.8:
                    group.append(b)
                    used[b] = True
            if len(group) < cfg.dev_text_min_letters:
                continue
            gxs = [cands[j][1] for j in group]
            heights = np.array([cands[j][3] for j in group], dtype=float)
            span = (max(gxs) - min(gxs)) / roi_w
            h_cv = heights.std() / max(heights.mean(), 1.0)
            if span >= cfg.dev_text_min_span and h_cv < cfg.dev_text_height_cv:
                logo.update(cands[j][0] for j in group)
        return logo

    # ── alternative: edge-boundary detector (rims only) ─────────────────────────
    def _edge_mask(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        """
        Detect strong edges (chunk boundaries) inside the ROI.
        Pre-blur suppresses texture noise; only high-gradient transitions survive.
        Edges are closed + filled to get solid unblended regions. Only catches the
        chunk rim, not its body — kept as an alternative to the deviation detector.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        erode_px = self.config.canny_roi_erode
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        interior_mask = cv2.erode(roi_mask, erode_kernel, iterations=1)

        masked = cv2.bitwise_and(blurred, roi_mask)
        edges = cv2.Canny(masked, self.config.canny_lo, self.config.canny_hi)
        edges = cv2.bitwise_and(edges, interior_mask)

        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        close_px = self.config.canny_close
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roi_area = int((roi_mask > 0).sum())
        filled = np.zeros_like(edges)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.canny_min_area or area > roi_area * 0.25:
                continue
            cv2.drawContours(filled, [cnt], -1, 255, thickness=cv2.FILLED)
        return self._fill_holes(filled)

    @staticmethod
    def _fill_holes(bw: np.ndarray) -> np.ndarray:
        """Fill enclosed background holes in a binary mask (flood-fill from a corner)."""
        h, w = bw.shape
        ff = bw.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(ff, flood_mask, (0, 0), 255)
        return bw | cv2.bitwise_not(ff)
