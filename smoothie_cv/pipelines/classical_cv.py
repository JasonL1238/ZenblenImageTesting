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

        chunk_detector = "canny"
        chunk_yolo_input = getattr(self.config, "chunk_yolo_input", "full_filter")

        if self.config.classical_method == "canny":
            roi = crop_to_roi(image, roi_mask)
            unblended = self._edge_mask(roi.image, roi.mask)
            unblended = cv2.bitwise_and(unblended, roi.mask)
            unblended = paste_mask(unblended, roi)
        else:
            # YOLO-chunk primary (full_filter or roi_crop), classical fallback.
            # Logo suppress runs inside the classical adapter only.
            from smoothie_cv.detection.chunk import detect_chunk
            unblended, chunk_detector = detect_chunk(
                image, roi_mask, self.config,
            )

        # Path 5: gated below-ROI cream band — classical-only. Do not layer
        # classical cream recovery on top of a successful YOLO chunk mask.
        if chunk_detector == "classical" and self.config.dev_botband_enable:
            band = self._bottom_cream_band(image, roi_mask)
            if band is not None and band.any():
                roi_mask = cv2.bitwise_or(roi_mask, band)
                unblended = cv2.bitwise_or(unblended, band)

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
                "chunk_detector": chunk_detector,
                "chunk_yolo_input": chunk_yolo_input,
            },
        )

    # ── Path 5: below-ROI cream-on-gasket band ──────────────────────────────────
    def _bottom_cream_band(self, image: np.ndarray, roi_mask: np.ndarray) -> np.ndarray | None:
        """Detect a thin unblended cream layer sitting on the gasket just BELOW the
        ROI. Returns a full-frame mask of the band, or None.

        Scans the central columns from y_bot downward for a contiguous run of bright,
        slightly-warm low-chroma rows (cream) bounded below by the dark gasket. The
        chroma window is the discriminator: gray hardware (ch≈0), glare (ch≈4–5) and
        dark shadow (ch≈1–5, dark L) are excluded; only warm off-white cream passes.
        """
        cfg = self.config
        ys, xs = np.where(roi_mask > 0)
        if ys.size < 100:
            return None
        y_top, y_bot = int(ys.min()), int(ys.max())
        x_lo, x_hi = int(xs.min()), int(xs.max())
        roi_h = max(y_bot - y_top, 1)
        H = image.shape[0]

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]
        chroma = np.sqrt((lab[:, :, 1] - 128.0) ** 2 + (lab[:, :, 2] - 128.0) ** 2)

        # body reference band (mid-cup) — same gates as Path 4
        rt = int(y_top + 0.35 * roi_h)
        rb = int(y_top + 0.55 * roi_h)
        ref_sel = (roi_mask > 0)[rt:rb + 1, :]
        if ref_sel.sum() < 20:
            return None
        body_L = float(np.median(L[rt:rb + 1, :][ref_sel]))
        body_ch = float(np.median(chroma[rt:rb + 1, :][ref_sel]))
        if body_L < cfg.dev_bot_min_body_L or body_ch < cfg.dev_bot_min_body_chroma:
            return None

        cl = x_lo + int(cfg.dev_botband_inset * (x_hi - x_lo))
        chi = x_hi - int(cfg.dev_botband_inset * (x_hi - x_lo))
        if chi <= cl:
            return None
        dark_thr = cfg.dev_botband_dark_drop * body_L
        max_ext = int(cfg.dev_botband_max_ext_frac * roi_h)

        band_rows: list[int] = []
        gasket = False
        band_done = False
        for y in range(y_bot + 1, min(H, y_bot + max_ext + 1)):
            rl = float(np.median(L[y, cl:chi + 1]))
            rc = float(np.median(chroma[y, cl:chi + 1]))
            if rl < dark_thr:               # hit the dark gasket → band is bounded below
                gasket = True
                break
            is_cream = rc <= cfg.dev_botband_chroma_hi and rl >= cfg.dev_botband_L_lo
            if is_cream and not band_done:
                band_rows.append(y)
            elif band_rows:                 # band ended; keep scanning for the gasket
                band_done = True

        if not (gasket and len(band_rows) >= cfg.dev_botband_min_h):
            return None
        band_L = float(np.median([np.median(L[y, cl:chi + 1]) for y in band_rows]))
        band_ch = float(np.median([np.median(chroma[y, cl:chi + 1]) for y in band_rows]))
        if not (cfg.dev_botband_chroma_lo <= band_ch <= cfg.dev_botband_chroma_hi):
            return None
        if not (cfg.dev_botband_L_lo <= band_L <= cfg.dev_botband_L_hi):
            return None

        # build the band mask: cream-like pixels (bright + low-chroma) across the cup
        # width, over the band rows. Restricting to the cream signature avoids grabbing
        # the surrounding holder/gasket pixels at those rows.
        out = np.zeros(roi_mask.shape, np.uint8)
        y0, y1 = band_rows[0], band_rows[-1]
        sub_L = L[y0:y1 + 1, x_lo:x_hi + 1]
        sub_ch = chroma[y0:y1 + 1, x_lo:x_hi + 1]
        cream = (sub_ch <= cfg.dev_botband_chroma_hi) & (sub_L >= cfg.dev_botband_L_lo) \
            & (sub_L <= cfg.dev_botband_L_hi)
        out[y0:y1 + 1, x_lo:x_hi + 1][cream] = 255
        return out

    # ── primary: colour-agnostic local-deviation detector ──────────────────────
    def _deviation_mask(self, image: np.ndarray, roi_mask: np.ndarray,
                        logo_mask: np.ndarray | None = None) -> np.ndarray:
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

        # ROI bounds — needed for the bright-neutral bottom exemption and foam cut.
        ys, xs = np.where(roi_mask > 0)
        y_top, y_bot = int(ys.min()), int(ys.max())
        x_left, x_right = int(xs.min()), int(xs.max())
        roi_h = max(y_bot - y_top, 1)
        roi_w = max(x_right - x_left, 1)
        foam_cut = int(y_top + cfg.dev_foam_frac * roi_h)
        exempt_y = (int(y_top + (1.0 - cfg.dev_bright_bot_exempt_frac) * roi_h)
                    if cfg.dev_bright_bot_exempt_frac > 0 else y_bot + 1)

        L, A, B = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
        chroma = np.sqrt((A - 128.0) ** 2 + (B - 128.0) ** 2)

        # masked large blur → local base colour (avoids outside-contour black bleed).
        # TWO passes: printed logo text is bright enough to pull the pass-1 base UP
        # in its whole K-neighbourhood, so ordinary smoothie between/inside letters
        # reads "darker than base" (the print's counter-shadow) and letter pixels
        # read less bright than they are. Pass 2 re-estimates the base from
        # smoothie pixels ONLY — print-signature pixels are excluded from the blur
        # exactly like outside-ROI pixels. The bottom zone keeps pass-1 behaviour
        # (cream masses are bright+neutral; their handling is the exemption's job).
        K = cfg.dev_blur_kernel | 1  # force odd
        def _masked_base(weight: np.ndarray) -> np.ndarray:
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
        if vals.size == 0:
            return np.zeros(roi_mask.shape, np.uint8)

        thr = float(vals.mean() + cfg.dev_k_sigma * vals.std())
        thr = max(thr, cfg.dev_min_delta_e)  # floor: ignore trivially small deviations
        raw = ((dE >= thr) & mi).astype(np.uint8) * 255

        # drop specular glare: very bright + low chroma (LAB a,b centred at 128)
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
        # Exempt the bottom zone from bright-neutral suppression: cream/pale unblended
        # masses at the cup bottom are bright+neutral vs their local base (K=121 mixes
        # them with the dark smoothie above → they look "logo-like").  The logo never
        # appears in the last dev_bright_bot_exempt_frac of the cup, so this exemption
        # recovers bottom chunks without touching the logo zone.
        bright_neutral[exempt_y:, :] = False
        raw[bright_neutral] = 0

        # despeckle (small open only). No close — keep logo letters as separate
        # components so the text-line detector can recognise their arrangement.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
        clean[:foam_cut, :] = 0

        # print halo: the bright-neutral suppression removes printed letters from the
        # map, but the K=121 base around them stays pulled up by their brightness, so
        # adjacent smoothie reads "darker than base" (print's counter-shadow). Dark
        # components living mostly inside this halo are rejected in the loop below.
        d = cfg.dev_dark_print_adj_dilate
        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1))
        # CLOSE first: letter counters and inter-letter gaps (the "n" counter etc.)
        # are enclosed by strokes and belong to the print footprint; then dilate
        # outward to cover the base-contamination fringe.
        print_halo = cv2.dilate(
            cv2.morphologyEx(bright_neutral.astype(np.uint8), cv2.MORPH_CLOSE, ck), ck) > 0

        # connected components → classify the "zenblen" logo (a horizontal row of
        # similar-height marks spanning a wide extent) and exclude it, then keep
        # the remaining compact, solid blobs as chunks.
        n, labels, stats, cents = cv2.connectedComponentsWithStats(clean, connectivity=8)
        logo_labels = self._logo_text_labels(stats, cents, n, roi_h, roi_w)

        # logo band: the vertical/horizontal extent of the CONFIRMED wordmark, with
        # a margin proportional to the letter height. Accepted components whose
        # centroid lands inside this band are print footprint (a counter-shadow blob
        # or a stray letter that didn't join the group), not chunks — rejected below.
        # Only defined when a wordmark was actually confirmed → never fires otherwise.
        logo_band = None
        if cfg.dev_logo_band_suppress and logo_labels:
            ll = list(logo_labels)
            tops = [stats[i, cv2.CC_STAT_TOP] for i in ll]
            bots = [stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in ll]
            lefts = [stats[i, cv2.CC_STAT_LEFT] for i in ll]
            rights = [stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in ll]
            my = cfg.dev_logo_band_margin_frac * float(
                np.median([stats[i, cv2.CC_STAT_HEIGHT] for i in ll]))
            med_letter_area = float(np.median([stats[i, cv2.CC_STAT_AREA] for i in ll]))
            logo_band = (min(tops) - my, max(bots) + my, min(lefts), max(rights),
                         cfg.dev_logo_band_max_area_mult * med_letter_area)

        # per-pixel deviation vector from local base (for directional growth below)
        dev = lab - base

        # doubly-eroded interior: used to reject rim/wall-junction artifacts in the
        # meniscus band (they hug the ROI contour; real surface lumps are interior).
        ek2 = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px * 4 + 1, erode_px * 4 + 1))
        interior2 = cv2.erode(roi_mask, ek2, iterations=1) > 0

        roi_area = int((roi_mask > 0).sum())
        # bottom exempt line for the component-level print/glare rejection below —
        # same zone as the pixel-level bright-neutral exemption (cream masses at the
        # cup bottom share the bright+desaturating signature of the logo).
        comp_exempt_y = int(y_top + (1.0 - cfg.dev_bright_bot_exempt_frac) * roi_h)
        out = np.zeros(roi_mask.shape, np.uint8)
        grow_seed_masks: list[np.ndarray] = []  # seeds large enough to grow
        for lab_i in range(1, n):
            if lab_i in logo_labels:
                continue
            area = int(stats[lab_i, cv2.CC_STAT_AREA])
            # confirmed-wordmark band suppression (see logo_band above): a
            # letter-sized component whose centroid lands inside the wordmark band
            # is print footprint, not a chunk. Larger masses that merely overlap the
            # band are real chunks and are spared by the area ceiling.
            if logo_band is not None and area <= logo_band[4]:
                cy_i, cx_i = cents[lab_i][1], cents[lab_i][0]
                if (logo_band[0] <= cy_i <= logo_band[1]
                        and logo_band[2] <= cx_i <= logo_band[3]):
                    continue
            # trained-logo-mask suppression (ADDITIVE; see detect_logo). A
            # component whose pixels fall mostly inside the learned wordmark mask
            # is print footprint — reject it. Fraction-of-component (not IoU): a
            # real chunk grazing a letter keeps most of its mass outside the tight
            # mask and survives. Catches the CLIPPED-wordmark FPs the classical
            # text-line detector can't confirm. Gates ALL paths (compact/dark/chroma).
            if logo_mask is not None:
                comp = labels == lab_i
                inside = int((comp & (logo_mask > 0)).sum())
                if inside / max(area, 1) >= cfg.dev_logo_yolo_overlap:
                    continue
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

            # component-level print/glare rejection: much BRIGHTER than the local
            # base and DESATURATING vs it — the logo/glare signature (a real chunk
            # is darker or more saturated). Catches lone clipped logo letters that
            # defeat both the pixel-level bright-neutral rule (absolute chroma rides
            # above dev_bright_chroma on saturated smoothies) and the text-line
            # detector (<3 letters in frame). Bottom zone exempt: cream masses.
            if (mean_dL > cfg.dev_bright_dL
                    and mean_dC < cfg.dev_comp_bright_dC_max
                    and cents[lab_i][1] < comp_exempt_y):
                continue

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
            halo_frac = float(print_halo[comp_mask].mean())
            dark = (mean_dL <= cfg.dev_dark_dL
                    and solidity >= cfg.dev_dark_min_solidity
                    and extent >= cfg.dev_dark_min_extent
                    # print counter-shadow rejection + meniscus-band position gate
                    # (real dark chunks sit at y_frac ≥ 0.18 and outside print)
                    and halo_frac < cfg.dev_dark_print_adj_frac
                    and cents[lab_i][1] >= y_top + cfg.dev_relaxed_top_frac * roi_h)
            # Path 3 — chroma deviation: hue-similar chunks (orange-on-yellow, amber
            # flecks) are NOT darker than base so Path 2 misses them, but they are more
            # SATURATED than base. Glare/highlights/logo desaturate (mean_dC < 0), so
            # this gate spares them — same precision principle as the darkness gate.
            # brightness ceiling: warm print on pale cups is MORE saturated than the
            # body (dC>0, defeating the desaturation assumption) but backlit-bright
            # (ΔL +11…+14); pigmented fruit never gains brightness (ΔL −8…+1).
            in_band = cents[lab_i][1] < y_top + cfg.dev_relaxed_top_frac * roi_h
            chromatic = (mean_dC >= cfg.dev_chroma_dC
                         and mean_dL <= cfg.dev_chroma_dL_max
                         and solidity >= cfg.dev_dark_min_solidity
                         and extent >= cfg.dev_dark_min_extent
                         # meniscus band: rim/wall-junction slivers hug the contour;
                         # a real surface lump is deeply interior
                         and (not in_band or float(interior2[comp_mask].mean())
                              >= cfg.dev_chroma_band_interior))
            # the colour-cued paths (dark/chroma) may be elongated streaks → relaxed
            # aspect; logo strokes are excluded by neutrality (mean_dC<0, mean_dL>0).
            colour_cued = (dark or chromatic) and aspect <= cfg.dev_relaxed_aspect_hi

            if compact or colour_cued:
                # top-corner logo suppression (see dev_logo_corner_* in config): a
                # clipped-wordmark fragment that defeats the text-line detector lands
                # in the top-left/top-right CORNER of the ROI, a zone real chunks avoid
                # (they never sit above y_frac≈0.25 and cluster centrally). Veto an
                # otherwise-accepted component whose centroid is in the top band AND
                # near a vertical edge. Non-destructive: drops one detection only.
                # ONLY when NO wordmark was confirmed (not logo_labels): with a
                # confirmed wordmark the existing logo_band suppression already handles
                # the letters via its area ceiling AND correctly spares a real dark
                # chunk that happens to sit in the top corner (measured: 8343d981,
                # ac4eac46 — dark high-solidity chunks in the top-left, position-
                # identical to logo fragments, so ONLY the wordmark-confirmed gating
                # separates them). The clipped case this rule targets always has an
                # UNconfirmed wordmark (<3 letters / short span → logo_labels empty).
                if (cfg.dev_logo_corner_suppress and not logo_labels and not (
                        cfg.dev_logo_corner_compact_only and colour_cued)):
                    cy_f = (cents[lab_i][1] - y_top) / roi_h
                    cx_f = (cents[lab_i][0] - x_left) / roi_w
                    edge_d = min(cx_f, 1.0 - cx_f)
                    if (cy_f <= cfg.dev_logo_corner_y_max
                            and edge_d <= cfg.dev_logo_corner_edge_max):
                        continue
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
        # a grown pixel must deviate in raw magnitude too (not just project onto the
        # seed direction): low-contrast haze/condensation weakly aligns with the seed
        # direction everywhere, so projection alone bleeds across flat smoothie. Gate
        # on dE >= a fraction of the per-image seed threshold to keep growth on the
        # chunk's genuine fading margin.
        grow_min_dE = cfg.dev_grow_min_dE_frac * thr
        magnitude_ok = dE >= grow_min_dE
        for seed_mask in grow_seed_masks:
            sig = dev[seed_mask].mean(axis=0)        # chunk's mean deviation direction
            norm = float(np.linalg.norm(sig))
            if norm < 1e-3:
                continue
            proj = (dev * (sig / norm)).sum(axis=2)  # how far each pixel deviates that way
            field = ((proj >= cfg.dev_grow_proj_thr) & magnitude_ok & mi).astype(np.uint8) * 255
            field[excluded] = 0
            field[:foam_cut, :] = 0
            seed = seed_mask.astype(np.uint8) * 255
            out = cv2.bitwise_or(out, self._reconstruct(seed, field, cfg.dev_grow_max_iter))

        # ── Path 7: chroma-plane deviation ────────────────────────────────────
        # Hue-similar lumps on pale bodies deviate almost purely in chroma with
        # total ΔE below the global floor, so the dE map never sees them. Threshold
        # the dC map with its own adaptive threshold; precision comes from the same
        # relative-colour gates as the chroma path (saturated but NOT brighter than
        # base — print/glare brighten or desaturate) plus the shared exclusions.
        if cfg.dev_chroma_plane_enable:
            dc_vals = dC[mi]
            dc_thr = max(float(dc_vals.mean() + cfg.dev_chroma_plane_k_sigma
                               * dc_vals.std()), cfg.dev_chroma_plane_min_dC)
            dc_raw = ((dC >= dc_thr) & mi).astype(np.uint8) * 255
            dc_raw[glare | bright_neutral] = 0
            dc_raw[:foam_cut, :] = 0
            dc_clean = cv2.morphologyEx(dc_raw, cv2.MORPH_OPEN, k)
            n_c, labels_c, stats_c, cents_c = cv2.connectedComponentsWithStats(
                dc_clean, connectivity=8)
            for li in range(1, n_c):
                area_c = int(stats_c[li, cv2.CC_STAT_AREA])
                if (area_c < cfg.dev_chroma_plane_min_area
                        or area_c > roi_area * cfg.dev_max_area_frac):
                    continue
                cm = labels_c == li
                if float(dL[cm].mean()) > cfg.dev_chroma_dL_max:
                    continue          # brighter than base → print/glare, not pigment
                if cents_c[li][1] < y_top + cfg.dev_relaxed_top_frac * roi_h:
                    continue          # meniscus band
                if float(print_halo[cm].mean()) >= cfg.dev_dark_print_adj_frac:
                    continue          # inside the print footprint
                out[cm] = 255

        # ── Path 4: bottom absolute-chroma gate ──────────────────────────────
        # Cups with unblended cream/white masses at the cup bottom have nearly
        # ZERO chroma in the last few ROI rows (ch≈5–10).  K=121 adapts to large
        # masses (ΔE≈0), making them invisible to paths 1–3.
        #
        # Detection: if the MEDIAN absolute chroma of the last dev_bot_n_rows rows
        # is ≤ dev_bot_abs_chroma_max, flag those rows as a cream-chunk region.
        #
        # Precision gates:
        #   • body_L ≥ 95: dark cups (maroon/dark-red) naturally lose chroma at
        #     the hardware gasket — excluding them avoids that FP.
        #   • body_chroma ≥ 22: skips pale/yellow cups whose body chroma is already
        #     low, where the absolute floor has no discriminative power.
        #   • bot_med_ch ≤ 11 (not a relative drop): a cream mass drops to ch≈5–10
        #     uniformly; the gasket transition on any cup also produces a 1–2 row
        #     dip to ch≈5 but the OTHER rows stay ≥12, so the 6-row MEDIAN stays
        #     above 11. Using absolute chroma sidesteps the misleading "large drop"
        #     that a single gasket row can create when the rest of the bottom is
        #     still chromatic.
        if cfg.dev_bot_n_rows > 0:
            bot_t = max(y_top, y_bot - cfg.dev_bot_n_rows + 1)
            bot_sel = (roi_mask > 0)[bot_t:y_bot + 1, :]
            bot_px = lab[bot_t:y_bot + 1, :][bot_sel]
            if bot_px.shape[0] >= 20:
                bot_ch_vals = np.sqrt((bot_px[:, 1] - 128.0) ** 2
                                      + (bot_px[:, 2] - 128.0) ** 2)
                bot_med_ch = float(np.median(bot_ch_vals))
                ref_t2 = int(y_top + 0.35 * roi_h)
                ref_b2 = int(y_top + 0.55 * roi_h)
                ref_px2 = lab[ref_t2:ref_b2 + 1, :][
                    (roi_mask > 0)[ref_t2:ref_b2 + 1, :]]
                if ref_px2.shape[0] >= 20:
                    ref_med_L = float(np.median(ref_px2[:, 0]))
                    ref_med_ch = float(np.median(
                        np.sqrt((ref_px2[:, 1] - 128.0) ** 2
                                + (ref_px2[:, 2] - 128.0) ** 2)))
                    if (ref_med_L >= cfg.dev_bot_min_body_L
                            and ref_med_ch >= cfg.dev_bot_min_body_chroma
                            and bot_med_ch <= cfg.dev_bot_abs_chroma_max):
                        bot_mask = (roi_mask > 0).copy()
                        bot_mask[:bot_t, :] = False
                        out[bot_mask] = 255

        # ── reference-band deviation pass ─────────────────────────────────────
        # The K=121 Gaussian adapts to any region larger than ~60px radius, making
        # large monochromatic masses (e.g. a pale cream layer at the bottom of a dark
        # smoothie) invisible to the local ΔE map (local ΔE ≈ 0).
        #
        # Fix: compare pixels in the LOWER ZONE (below target_top_frac of the ROI)
        # against the mean colour of a REFERENCE BAND just above the lower zone
        # (ref_top_frac .. ref_bot_frac). A cream mass creates a sharp colour jump;
        # natural gradient changes gradually so reference ≈ lower-zone colour.
        # The area gate (dev_global_min_area) eliminates logo letters and glare.
        if cfg.dev_global_enable:
            ref_top_y = int(y_top + cfg.dev_global_ref_top_frac * roi_h)
            ref_bot_y = int(y_top + cfg.dev_global_ref_bot_frac * roi_h)
            target_top_y = int(y_top + cfg.dev_global_target_top_frac * roi_h)

            ref_sel = mi.copy()
            ref_sel[:ref_top_y, :] = False
            ref_sel[ref_bot_y:, :] = False
            ref_pixels = lab[ref_sel]
            if ref_pixels.shape[0] >= 50:
                ref_mean = ref_pixels.mean(axis=0)
                dE_ref = np.sqrt(((lab - ref_mean) ** 2).sum(axis=2))

                # colour gate, TWO branches (union — one physical mass often has both):
                # 1. NEUTRAL: unblended cream/neutral masses are less saturated than
                #    the coloured smoothie body.  Lighter-pink lower zones (natural
                #    gradient) keep the same hue, so their chroma stays close to the
                #    reference.  Require pixels to be ≥ dev_global_chroma_drop units
                #    LESS chromatic than the reference band mean.
                # 2. HUE-SHIFTED: a saturated mass of a DIFFERENT colour family
                #    (yellow banana/mango on a red smoothie, 50e294) keeps chroma
                #    similar to the body — the neutral branch never sees it — but
                #    sits far from the reference in ab-plane ANGLE. Natural bottom
                #    gradients lighten/darken WITHIN the body's hue family (small
                #    angle), so a large hue swing is a real colour jump. The chroma
                #    floor excludes near-neutral pixels whose hue is numerically
                #    unstable (glare, shadow). Measured on the 92-set: only genuine
                #    bottom masses (50e294, 749a) produce a bottom-attached hue
                #    component anywhere in θ 25–50°; every clean cup produces none.
                A_ch = lab[:, :, 1]
                B_ch = lab[:, :, 2]
                chroma_px = np.sqrt((A_ch - 128.0) ** 2 + (B_ch - 128.0) ** 2)
                ref_chroma = float(np.sqrt((ref_mean[1] - 128.0) ** 2
                                          + (ref_mean[2] - 128.0) ** 2))
                neutral_vs_ref = (ref_chroma - chroma_px) >= cfg.dev_global_chroma_drop

                ref_hue = np.degrees(np.arctan2(ref_mean[2] - 128.0,
                                                ref_mean[1] - 128.0))
                hue_px = np.degrees(np.arctan2(B_ch - 128.0, A_ch - 128.0))
                hue_diff = np.abs(hue_px - ref_hue)
                hue_diff = np.minimum(hue_diff, 360.0 - hue_diff)
                hue_vs_ref = ((hue_diff >= cfg.dev_global_hue_deg)
                              & (chroma_px >= cfg.dev_global_hue_min_chroma))

                colour_jump = neutral_vs_ref | hue_vs_ref

                target_sel = mi.copy()
                target_sel[:target_top_y, :] = False
                global_raw = (target_sel & (dE_ref >= cfg.dev_global_thr)
                              & colour_jump).astype(np.uint8) * 255
                global_raw[:foam_cut, :] = 0

                k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                global_clean = cv2.morphologyEx(global_raw, cv2.MORPH_CLOSE, k5)
                global_clean = cv2.morphologyEx(global_clean, cv2.MORPH_OPEN, k5)
                n_g, labels_g, stats_g, _ = cv2.connectedComponentsWithStats(
                    global_clean, connectivity=8)
                # full-ROI cream field for reconstruction: the accepted mass often
                # extends into the dev_roi_erode boundary band (cream rests against
                # the cup wall/gasket), which mi excludes. dE_ref has no masked-blur
                # boundary artifact, so growing into the un-eroded ROI is sound; the
                # L floor keeps the dark gasket edge from joining the mass.
                cream_field = ((roi_mask > 0) & (dE_ref >= cfg.dev_global_thr)
                               & colour_jump
                               & (lab[:, :, 0] >= cfg.dev_botband_L_lo))
                cream_field[:max(target_top_y, foam_cut), :] = False
                cream_field_u8 = cream_field.astype(np.uint8) * 255
                # bottom-attachment gate: cream is heavy and rests on the cup
                # bottom/gasket; the diffuse backlit glare glow floats mid-low cup.
                # Require the component to reach near the ROI's last row.
                attach_y = y_bot - int(cfg.dev_global_bot_attach_frac * roi_h)
                for li in range(1, n_g):
                    comp_bot = (int(stats_g[li, cv2.CC_STAT_TOP])
                                + int(stats_g[li, cv2.CC_STAT_HEIGHT]) - 1)
                    if (int(stats_g[li, cv2.CC_STAT_AREA]) >= cfg.dev_global_min_area
                            and comp_bot >= attach_y):
                        comp = ((labels_g == li).astype(np.uint8)) * 255
                        out = cv2.bitwise_or(
                            out, self._reconstruct(comp, cream_field_u8,
                                                   cfg.dev_global_grow_max_iter))

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
