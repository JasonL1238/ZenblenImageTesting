"""
Central config for the smoothie blendedness pipeline.

Priority (highest → lowest):
  CLI flags  >  config.yaml  >  dataclass defaults
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


@dataclass
class Config:
    # --- pass/fail ---
    threshold: float = 0.90          # blend_score >= threshold → passed

    # --- classical CV ---
    # Primary chunk detector: "deviation" (colour-agnostic LAB local-deviation) or
    # "canny" (edge-boundary, rims only).
    classical_method: str = "deviation"

    # local-deviation detector (colour-agnostic)
    dev_blur_kernel: int = 121     # px Gaussian kernel for the local base colour
    dev_roi_erode: int = 14        # px erode ROI inward (skip jar-wall/contour edge)
    dev_k_sigma: float = 2.5       # adaptive threshold = mean + k·σ of ΔE in ROI
    dev_min_delta_e: float = 12.0  # floor on ΔE so trivially small deviations never fire
                                   # (12 not 16: hue-similar chunks — orange-on-yellow —
                                   #  deviate <16 ΔE; precision held by the colour/dark gates)
    dev_foam_frac: float = 0.16    # exclude this fraction of ROI height at the top (foam/rim).
                                   # 0.16 not 0.12: the meniscus shadow/highlight band just
                                   # below the old cut fired the relaxed colour-cued paths;
                                   # real chunks sit at y_frac >= 0.18, so 0.16 spares them.
    dev_min_area: int = 200        # min blob area (px) to count as a chunk (compact path)
    dev_compact_min_delta_e: float = 16.0  # compact path carries NO colour cue, so it keeps
                                           # the original ΔE floor; only the colour-cued paths
                                           # exploit the lower dev_min_delta_e. Blocks bright
                                           # glare blobs the lowered global floor would admit.
    dev_max_area_frac: float = 0.30  # max blob area as a fraction of ROI
    # shape gate — real chunks are compact blobs; logo text is thin / extreme-aspect
    dev_min_solidity: float = 0.62   # reject low-solidity letter clusters (logo)
    dev_min_extent: float = 0.40     # reject sparse/stringy blobs
    dev_aspect_lo: float = 0.4       # reject very wide or very tall bars (logo word/strokes)
    dev_aspect_hi: float = 2.6
    # dark-deviation path — recovers subtle chunks (dark arcs) the compact gate misses.
    # Darkness preserves precision: logo/glare are brighter than base, never darker.
    dev_dark_dL: float = -15.0        # component mean ΔL must be at least this dark vs base
    dev_dark_min_solidity: float = 0.30
    dev_dark_min_extent: float = 0.18
    # chroma-deviation path — recovers hue-similar chunks (orange-on-yellow, amber
    # flecks) that are NOT darker than base, so the dark path misses them. Keyed on
    # the component being more SATURATED than the local base (mean ΔchromaC). This
    # preserves precision the same way darkness does: glare/highlights/logo DESATURATE
    # (ΔchromaC < 0), so they never pass — only genuinely coloured lumps do.
    dev_chroma_dC: float = 8.0        # component mean ΔchromaC must exceed this (more saturated)
    dev_chroma_band_interior: float = 0.7  # a chroma-path component whose centroid is ABOVE the
                                           # dev_relaxed_top_frac line (meniscus band) must have
                                           # ≥ this fraction of its area inside a doubly-eroded
                                           # (2×dev_roi_erode) ROI interior. The rim/wall junction
                                           # produces saturated dark slivers HUGGING the contour
                                           # (measured 0.28 interior); a real lump floating on the
                                           # surface is surrounded by smoothie (measured 1.00).
                                           # Below the band no interiority is required.
    dev_chroma_dL_max: float = 5.0    # chroma-path brightness ceiling: the component must NOT
                                      # be brighter than the local base by more than this.
                                      # Pigmented fruit absorbs light — measured real chroma-path
                                      # chunks sit at mean ΔL −8…+1 — while warm backlit print
                                      # (logo fragments on pale/tan cups, where cream ink is MORE
                                      # saturated than the body, defeating the dC<0 assumption)
                                      # sits at ΔL +11…+14. Brightness is the discriminator.
    # relaxed shape gates shared by the dark + chroma paths (both already carry a
    # colour/darkness cue that logo text lacks, so shape can be looser than compact).
    dev_relaxed_min_area: int = 90    # min blob area for the colour-cued paths (< dev_min_area)
    dev_relaxed_top_frac: float = 0.18  # DARK-path components must have their centroid below
                                        # this fraction of ROI height. The meniscus SHADOW band
                                        # hangs just below the foam cut (0.16) and fires the dark
                                        # path when the ROI's top geometry shifts (YOLO vs SAM);
                                        # measured real dark chunks sit at y_frac ≥ 0.29 — same
                                        # finding that set the foam cut ("real chunks sit at
                                        # y_frac≥0.18"). NOT applied to the chroma path: a real
                                        # saturated lump can ride at the surface (f0b6a6d1), and
                                        # chroma-path print FPs are already killed by the
                                        # dev_chroma_dL_max brightness ceiling.
    # dark-path print-shadow rejection: pixel-level bright-neutral suppression removes
    # printed letters from the ΔE map, but the K=121 local base around them stays
    # PULLED UP by their brightness, so the ordinary smoothie between/around letters
    # measures as "darker than base" and fires the dark path (print's counter-shadow).
    # A dark component whose area mostly lies inside the dilated print (bright-neutral)
    # halo is that artifact, not a chunk — real dark chunks don't live inside print.
    dev_dark_print_adj_dilate: int = 7     # px dilation of the bright-neutral mask → print halo
    dev_dark_print_adj_frac: float = 0.5   # reject dark components with ≥ this fraction of
                                           # their area inside the halo
    dev_relaxed_aspect_hi: float = 5.0  # colour-cued chunks may be elongated streaks; logo
                                        # strokes are excluded by neutrality, not aspect
    # hysteresis region-grow: the strict gates above find the high-contrast CORE of a
    # chunk; its lower-contrast margin (fading toward smoothie colour) sits below the
    # seed threshold and is dropped, so only PART of the chunk is masked. Grow each
    # accepted seed into contiguous pixels above a LOWER ΔE threshold to recover the
    # full extent. Seeded growth keeps precision: faint texture/glare with no seed of
    # its own never grows. (Fixes "got part of the chunk but not the whole chunk".)
    # Growth is DIRECTIONAL: a chunk's faint margin/tail (fading toward smoothie colour)
    # deviates from the local base in the SAME colour direction as the chunk core (e.g.
    # redder + darker), just with smaller magnitude. So we grow each seed into contiguous
    # pixels whose deviation-from-base vector PROJECTS strongly onto the seed's mean
    # deviation direction — this captures the tail (same direction, low magnitude) that a
    # raw-ΔE-magnitude threshold drops, while uniform smoothie (no consistent direction)
    # and off-direction glints are excluded.
    dev_grow_proj_thr: float = 5.0      # min projection (LAB units) onto the chunk's
                                        # signature direction for a pixel to join. Lowered
                                        # 7->5 so a chunk's faint same-direction body/tail
                                        # (e.g. cf4d's amber lump, only ~20% covered before)
                                        # is reached; bleed is held off by dev_grow_min_dE_frac.
    dev_grow_min_dE_frac: float = 0.5   # a grown pixel must ALSO deviate in raw magnitude:
                                        # dE >= this * (per-image seed threshold). Projection
                                        # alone grows across low-contrast haze/condensation
                                        # (e.g. dd4d29) where noise weakly aligns with the seed
                                        # direction; requiring real deviation magnitude keeps
                                        # growth on the genuine fading margin, not flat smoothie.
    dev_grow_max_iter: int = 40         # bound growth to ~40 px from the seed (reach the
                                        # tail of larger lumps, but can't crawl across the cup)
    dev_grow_min_seed_area: int = 100   # only GROW seeds at least this big (a confident
                                        # chunk core worth completing). Tiny marginal seeds
                                        # (logo letter, rim glare, lone fleck) are kept as-is
                                        # but NOT amplified — growth must not flip a near-clean
                                        # smoothie to flagged on a borderline speck.
                                        # 200→100: with the component-level print/glare gates
                                        # (bright-desat, chroma brightness ceiling, print halo,
                                        # meniscus position) the seeds that reach growth are
                                        # far cleaner than when 200 was set; 100 lets a small
                                        # real chunk core (e.g. afdc6c3e's 102px lump) complete
                                        # to its true extent instead of stalling sub-flag.
    # global (reference-band) deviation — catches large monochromatic regions
    # (e.g. a pale cream mass at the bottom of a dark smoothie) that are invisible
    # to the K=121 local blur because the Gaussian spans the whole region and
    # adapts its base *to* the mass (ΔE≈0).
    #
    # Approach: compare pixels in the lower zone (below dev_global_target_top_frac)
    # against the mean colour of a REFERENCE BAND just above the lower zone
    # (dev_global_ref_top_frac .. dev_global_ref_bot_frac). A cream mass creates a
    # sharp colour jump from the reference to the lower zone; natural smoothie
    # gradient changes gradually so the reference band closely matches the lower
    # zone.  A large area gate (dev_global_min_area) excludes logo letters and
    # glare specks.
    #
    # Design decisions:
    #   - Reference band is placed at 45–62% ROI height — well within the dark
    #     smoothie body, safely above any bottom chunk (which is in the last ~25%).
    #   - Target zone starts at 62% — even if a chunk occupies the bottom 30% the
    #     reference band above it is still pure smoothie.
    #   - No locally_invisible gate: the lower-zone restriction + area gate are the
    #     primary precision controls; requiring low local-ΔE would silently drop
    #     the chunk boundary pixels (where K=121 straddles cream and dark smoothie).
    #   - Foam band is always excluded (foam_cut already applied to out).
    dev_global_enable: bool = True   # re-enabled: with ROIs that reach the true cup bottom
                                     # (YOLO-seg / gasket-complete masks), large cream masses
                                     # sit fully INSIDE the ROI where K=121 adapts to them —
                                     # paths 1–3 see ΔE≈0 and Path 4's last-rows gate can sit
                                     # on gasket pixels and miss. This pass segments the mass
                                     # body itself (additive; area-gated ≥1500px; does not
                                     # perturb the adaptive per-image threshold).
    dev_global_thr: float = 22.0            # ΔE from reference band to flag a pixel
    dev_global_chroma_drop: float = 12.0    # NEUTRAL branch: detected pixels must be ≥ this
                                            # many chroma units LESS saturated than the
                                            # reference band mean.  Keeps light-pink lower
                                            # zones (natural smoothie gradient, still reddish)
                                            # from firing; cream/neutral masses drop ≥ 20.
    # HUE branch (OR'd with the neutral branch): a saturated mass of a different
    # colour FAMILY (yellow on red, 50e294) keeps chroma similar to the body — the
    # neutral branch is blind to it — but swings the LAB ab-plane hue angle hard.
    # Natural gradients lighten within the body's hue family (small angle).
    # Margin measured on the 92-set: only genuine bottom masses (50e294, 749a)
    # produce a bottom-attached hue component at ANY θ in 25–50°; clean cups
    # produce zero, and component size varies <25% across that whole range.
    dev_global_hue_deg: float = 35.0        # min ab-plane hue-angle diff vs reference (°)
    dev_global_hue_min_chroma: float = 12.0 # pixel chroma floor — hue is numerically
                                            # unstable near neutral (glare/shadow px)
    dev_global_grow_max_iter: int = 512     # Path 6 reconstruction bound. NOT the shared
                                            # dev_grow_max_iter (=40): that bound protects
                                            # the loose directional field of paths 1–3; a
                                            # bottom mass spans the cup (~250px) and 40 px
                                            # of geodesic reach left 2/3 of 50e294's mass
                                            # uncovered. Path 6's field is already gated
                                            # (ΔE≥22 + colour jump + L floor + lower zone)
                                            # and only grows when a ≥1500px bottom-attached
                                            # seed exists — clean cups have no seed, so a
                                            # large bound cannot introduce churn.
    dev_global_min_area: int = 1500         # min connected-component area (px)
    dev_global_bot_attach_frac: float = 0.05  # a component must reach within this fraction
                                              # of ROI height of the ROI's LAST row. Unblended
                                              # cream is heavy — it sinks and RESTS on the cup
                                              # bottom/gasket; the diffuse backlit glare glow
                                              # (condensation halo) floats mid-low cup, detached
                                              # from the bottom rows. Bottom attachment is the
                                              # physical discriminator between the two — without
                                              # it the pass flags the glow on many clean cups.
    dev_global_ref_top_frac: float = 0.45   # reference band top (fraction of ROI height)
    dev_global_ref_bot_frac: float = 0.62   # reference band bottom
    dev_global_target_top_frac: float = 0.62  # lower zone starts here

    # ── Path 7: chroma-plane deviation ────────────────────────────────────────
    # Hue-similar lumps on PALE bodies (soft mango spots on pale yellow, 636e83f4)
    # deviate almost purely in CHROMA (dC≈+8…10) with total ΔE below the global
    # floor (12), so they never form components in the dE map. Threshold the dC
    # map directly with its own adaptive threshold (mean + k·σ of dC inside the
    # ROI, floored). Precision reuses the established relative-colour gates: a
    # component must be saturated-not-bright (dev_chroma_dL_max ceiling — print
    # and glare desaturate or brighten), sit below the meniscus band, and clear
    # the same glare/bright-neutral/foam exclusions as every other path.
    # DISABLED — measured signal-floor dead end (2026-07-02, same conclusion as the
    # k_sigma experiments): on the 92-set it recovers 1 real cup (636e83f4, lumps
    # at dC +8.6…+9.7) but flips 4 audited-clean cups whose embossed-ridge /
    # glow-rim strips measure dC +7.7…+9.6 — ZERO separating margin at this
    # contrast. Keep the code for if a polarizer/diffuse-light upgrade lifts the
    # floor; do NOT enable and tune thresholds — there is no threshold to find.
    dev_chroma_plane_enable: bool = False
    dev_chroma_plane_k_sigma: float = 2.5   # adaptive: mean + k·σ of the dC map in ROI
    dev_chroma_plane_min_dC: float = 7.0    # floor on the threshold (LAB chroma units)
    dev_chroma_plane_min_area: int = 90     # same floor as the other colour-cued paths

    dev_glare_L: float = 240.0     # LAB L* above this + low chroma = specular glare
    dev_glare_chroma: float = 12.0
    # printed-logo / backlit-text exclusion: bright-vs-base + low chroma (cream text)
    dev_bright_dL: float = 25.0    # ΔL above local base to be considered "bright text"
    dev_bright_chroma: float = 22.0
    # COMPONENT-level print/glare rejection (complements the pixel-level rule above):
    # a component much brighter than the local base (mean ΔL > dev_bright_dL) that
    # also DESATURATES vs base (mean ΔchromaC below this) is printed logo text or
    # glare, never a chunk — real chunks are darker than base or more saturated.
    # This catches logo letters the pixel rule misses on saturated smoothies, where
    # cream print picks up enough absolute chroma to ride above dev_bright_chroma,
    # and the text-line detector can't fire because <3 letters are in frame (the
    # word clipped at the image edge, e.g. "ze…"). Keyed on colour RELATIVE to the
    # local base, so it holds across shades. The bottom exempt zone below still
    # applies — cream masses at the cup bottom are bright+desaturating too.
    dev_comp_bright_dC_max: float = 0.0
    # Cream/pale unblended masses at the CUP BOTTOM share the same bright+neutral
    # signature as the logo (high dL vs local base, low chroma) because the K=121
    # kernel mixes the cream with the dark smoothie above, making the base dark and
    # the cream pixels look "bright and neutral" to the exclusion filter.
    # The logo never appears in the last dev_bright_bot_exempt_frac of the cup height,
    # so exempting the bottom zone lets cream masses through while keeping the logo
    # suppression in place for the rest of the cup.
    dev_bright_bot_exempt_frac: float = 0.20  # bottom fraction of ROI exempt from
                                              # bright-neutral suppression (0 = no exemption)
    # Bottom neutrality check: cups with unblended cream masses at the bottom show a
    # sharp chroma drop in the last few rows.  K=121 adapts to large cream masses (ΔE≈0)
    # so the normal deviation paths miss them.  Instead, compare the median absolute
    # chroma of the very last rows to the mid-body reference band — a large drop flags
    # the bottom as "cream chunk present."  Only fires on chromatic (coloured) smoothies
    # (body chroma ≥ dev_bot_min_body_chroma) to skip pale/yellow cups.
    dev_bot_n_rows: int = 6                   # number of bottom rows to sample
    dev_bot_min_body_L: float = 95.0          # skip check on DARK cups (maroon/dark-red bodies
                                              # naturally lose chroma at the gasket transition,
                                              # mimicking a cream mass; this gates to LIGHT cups
                                              # only where very low bottom chroma is truly anomalous)
    dev_bot_min_body_chroma: float = 22.0     # skip check if body is too pale/neutral
    dev_bot_abs_chroma_max: float = 11.0      # flag if ABSOLUTE median chroma of the bottom rows
                                              # is below this — real cream/white masses drop to
                                              # ch≈5–10; a natural gasket gradient is still ≥12.
                                              # Using absolute chroma (not relative drop) avoids
                                              # flagging cups where the gasket produces a transient
                                              # 1–2 row dip that pulls the drop metric high even
                                              # though most rows are still chromatic.
    # ── Path 5: below-ROI cream-on-gasket band ───────────────────────────────
    # SAM often cuts the ROI cleanly ABOVE a thin unblended cream layer that sits
    # on the holder gasket (e.g. 749a). Path 4 never sees it (cream is below y_bot)
    # and global gasket-extend is too destabilising (perturbs the adaptive
    # threshold → 20 verdict flips on unrelated borderline cups, 0 target recovery).
    # Instead, scan the CENTRAL columns just below y_bot for a bright, slightly-warm
    # low-chroma band bounded below by the dark gasket — the cream signature. When
    # it fires, extend the ROI down over that band and flag it. Because the scan is
    # gated on the cream signature, only genuine-cream cups are touched; every other
    # ROI is byte-identical → zero churn (unlike global extend).
    #
    # The chroma WINDOW is the key discriminator: real cream is a warm off-white
    # (ch≈8–11), while the look-alikes below the ROI are near-neutral and excluded —
    # gray plastic holder clamp (db150e, ch≈0), specular glare (ch≈4–5), and dark
    # gasket-edge shadow (ch≈1–5, also dark L). The L window excludes those dark
    # shadows (L<100) and blown glare (L>145).
    dev_botband_enable: bool = True
    dev_botband_inset: float = 0.25        # horizontal inset per side (use central 1-2·inset)
    dev_botband_max_ext_frac: float = 0.08  # how far below y_bot to scan (fraction of ROI height)
    dev_botband_dark_drop: float = 0.55     # gasket row = central L < this · body_L
    dev_botband_min_h: int = 3              # min band thickness (rows) to fire
    dev_botband_chroma_lo: float = 7.0      # band median chroma floor (excludes gray/glare/shadow)
    dev_botband_chroma_hi: float = 12.0     # band median chroma ceiling (excludes chromatic smoothie)
    dev_botband_L_lo: float = 100.0         # band median L floor (excludes dark gasket shadow)
    dev_botband_L_hi: float = 145.0         # band median L ceiling (excludes blown specular glare)
    # logo-band suppression: once the text-line detector CONFIRMS a "zenblen"
    # wordmark (≥dev_text_min_letters aligned marks over a wide span), reject any
    # OTHER accepted component whose centroid falls inside that wordmark's
    # bounding band (± a margin proportional to the letter height). Rationale:
    # partial/curved framing leaves a counter-shadow blob or a stray letter that
    # did NOT join the group riding right alongside the confirmed letters — it is
    # part of the print footprint, not a chunk. This is CONDITIONAL on a confirmed
    # wordmark (never fires without one), so it cannot touch cups that have no
    # detected logo. It does NOT recover the "clipped wordmark" case where the
    # text-line detector never fires (<3 letters / span <dev_text_min_span) — that
    # needs a trained logo mask, not a classical rule.
    # A band component is only suppressed if it is LETTER-SIZED (area ≤ this
    # multiple of the confirmed wordmark's median letter area). The leaked print
    # blobs (counter-shadows / stray letters) measure ≤ a letter; a real chunk that
    # happens to overlap the wordmark band is a distinctly LARGER mass (measured:
    # leaked ≤274px vs real chunks 875/1752px at the same band). Size is the only
    # separator — the leaked blobs are dark+saturated, colour-identical to chunks.
    dev_logo_band_suppress: bool = True
    dev_logo_band_margin_frac: float = 0.5   # vertical margin above/below the band as
                                             # a fraction of the median letter height
    dev_logo_band_max_area_mult: float = 1.3  # suppress a band component only if its
                                              # area ≤ this × median letter area
    dev_letter_min_area: int = 60      # min component area to be a letter candidate
    dev_letter_h_lo: float = 0.025     # letter height range as fraction of ROI height
    dev_letter_h_hi: float = 0.20
    dev_text_min_letters: int = 3      # min aligned letters to call it a text line
    dev_text_min_span: float = 0.32    # min horizontal extent of the word (frac ROI width)
    dev_text_height_cv: float = 0.35   # max letter-height coeff. of variation (uniformity)

    # TOP-CORNER logo suppression (added 2026-07 from the 500-image labeling-disjoint
    # audit). The fixed rig prints the "zenblen" wordmark HIGH on the cup and it curves
    # toward the vertical edges, so a CLIPPED wordmark (partly out of frame / wrapping
    # around the cup → <3 visible letters or span <dev_text_min_span, which defeats the
    # text-line detector AND EAST) leaves 1–2 letter fragments in the top-LEFT / top-RIGHT
    # CORNERS of the ROI. Real chunks never sit above y_frac≈0.25 and cluster centrally
    # (measured: real-chunk y_frac ≥ 0.249, median 0.52; logo FPs median 0.27), so a
    # top-band + edge-proximity veto removes these fragments where no colour/size/stroke
    # rule can (SWT/MSER/topology measured to fully overlap — the compact dark/chroma
    # fragments are geometrically identical to small chunks). Complements — does NOT
    # replace — dev_logo_band_suppress, which needs a CONFIRMED ≥3-letter wordmark this
    # clipped case never produces. Applied only to a component that would otherwise be
    # ACCEPTED, so it vetoes one detection and never touches the ROI (non-destructive).
    # Audit result: catches 16/24 logo FPs (incl. 8/11 clipped-wordmark cases) at
    # 0/147 real-chunk loss. LIMIT: the ~8 non-corner FPs (esp. compact dark/chroma
    # letter fragments mid-frame) need a trained YOLO logo-mask class — not reachable
    # by any classical rule.
    dev_logo_corner_suppress: bool = True
    dev_logo_corner_y_max: float = 0.30      # veto only if centroid y_frac ≤ this (top band)
    dev_logo_corner_edge_max: float = 0.25   # AND min(x_frac,1-x_frac) ≤ this (near a
                                             # vertical edge) — the top-corner zone
    dev_logo_corner_compact_only: bool = False  # restrict veto to compact-path-only
                                                # components (protects a top-corner chunk
                                                # that also fires a colour path); measured
                                                # unnecessary (0 loss without it)

    # edge-boundary detector (alternative)
    canny_lo: int = 20
    canny_hi: int = 60
    canny_roi_erode: int = 14   # px to erode ROI inward before Canny to avoid jar-wall edges
    canny_close: int = 21       # px morph-close kernel to seal open chunk-boundary arcs
    canny_min_area: int = 150   # min contour area (px) to count as an unblended chunk

    # --- yellow ROI refinement ---
    # Fraction of min(H,W) to erode inward from the coarse geometry mask,
    # removing cup-edge plastic, reflections, and border glare.
    yellow_erode_scale: float = 0.018
    # Adaptive threshold: how many b* units below the center-crop median
    # a pixel can still be considered yellow smoothie.
    yellow_delta_b: float = 12.0
    # Hard cap on a* (0-centered): excludes pink/red contamination.
    yellow_a_max: float = 14.0
    # L* ceiling: pixels brighter than this are foam, glare, or specular — excluded.
    yellow_L_max: float = 220.0
    # Minimum LAB chroma (sqrt(a²+b²)) inside the ROI; rejects neutral metal/white.
    yellow_chroma_min: float = 6.0

    # --- container detection (ROI) ---
    # Order detectors are tried in. The fine-tuned YOLO-seg model is the priority
    # detector (trained on our own smoothie-only labels, reaches the true cup
    # bottom); classical colour-thresholding is the fallback, used only when YOLO
    # is unavailable or returns no plausible mask. SAM2 is registered but phased
    # out of the default order — force with prefer="sam" / --detector sam.
    detector_priority: list[str] = field(default_factory=lambda: ["yolo", "classical"])

    # --- YOLO-seg (container detection, PRIORITY) ---
    # Deployed weights. After retraining (training/train.py), promote the new run:
    #   cp runs/smoothie-seg/<run>/weights/best.pt checkpoints/yolo_smoothie_seg.pt
    # yolo_standard_seg.pt is the multi-mode-pipeline container detector (the
    # "standard" labeler mode), promoted 2026-07-14 over the older nano-v5
    # yolo_smoothie_seg.pt. NOTE: switching the container model shifts ROIs and
    # can re-flip borderline chunk verdicts (see the ROI-destabilises-threshold
    # note in CLAUDE.md) — always re-run scripts/validate_chunks.py after a change.
    yolo_weights: Path = field(
        default_factory=lambda: Path("checkpoints/yolo_standard_seg.pt"))

    # --- YOLO-seg LOGO suppression (chunk-detection FP filter, ADDITIVE) ---
    # A trained "zenblen"-wordmark seg model (training/train_multi.py --mode logo). When
    # dev_logo_yolo_suppress is on, detect_logo() produces a full-frame logo mask
    # and any accepted chunk component whose pixel-overlap with it is ≥
    # dev_logo_yolo_overlap is rejected as print footprint. This AUGMENTS the
    # classical _logo_text_labels()/band/corner rules — it targets the residual
    # CLIPPED-wordmark FPs the classical text-line detector can't confirm (too few
    # letters / short span). Default OFF: opt-in via --logo-yolo for A/B eval.
    dev_logo_yolo_suppress: bool = True  # ON by default (2026-07-14): the trained
                                         # logo mask is an additive FP filter — it only
                                         # removes chunk components landing on the wordmark,
                                         # targeting the clipped-wordmark FPs the classical
                                         # text-line detector can't confirm. A/B validated
                                         # 0 real-chunk loss before enabling.
    logo_weights: Path = field(
        default_factory=lambda: Path("checkpoints/yolo_logo_seg.pt"))
    logo_conf: float = 0.25          # instance confidence floor for the mask union
    # Reject a chunk component when this fraction of ITS pixels fall inside the
    # logo mask. Fraction-of-component (not IoU): a real chunk grazing a letter
    # keeps most of its mass outside the tight mask (low fraction → kept); a
    # clipped letter sits almost entirely inside (→ rejected). Raise toward
    # 0.6–0.7 if the A/B eval shows a real chunk lost to a letter it overlaps.
    dev_logo_yolo_overlap: float = 0.5

    # --- SAM2 (container detection, LEGACY/reference) ---
    sam_model: str = "sam2_hiera_tiny"   # tiny preferred for Jetson compatibility
    # Top-edge prior policy. The RAW SAM mask is the primary output; the
    # straight-line top prior (flatten_roi_top) is applied ONLY when the raw
    # mask's top edge is too jagged — i.e. a "weird" mask whose ragged rim would
    # otherwise drag the foam/meniscus band into the ROI and fire the chunk
    # detector. Flatten iff top_edge_roughness(raw) > sam_top_roughness_max.
    # 2.5 px: below this the raw top is clean enough to keep as-is; above it the
    # rim is squiggly enough to need straightening. (A blunt single-metric gate —
    # roughness can't perfectly separate every misfire, so ~2 faint foam FPs may
    # leak vs. always-flatten; the trade is that most masks keep their true,
    # un-straightened surface geometry.)
    sam_top_roughness_max: float = 2.5
    # Side-wall refinement: median-smooth the per-row left/right walls over this
    # fraction of the cup height to straighten ragged sides (logo-text scallops,
    # low-confidence jitter on dark fills) that otherwise drag thin dark slivers
    # into the ROI and misfire the chunk detector. Robust median => a clean wall is
    # unchanged and it never extends past the true wall. 0 disables.
    sam_side_refine_win: float = 0.06
    # Fixed-rig bottom prior (DISABLED by default — see why below). On dark fills
    # SAM stops mid-cup where the smoothie blends into the shadowed holder, leaving
    # a big bottom chunk outside the ROI (the cup then scores falsely clean, e.g.
    # 50e294/749a). `extend_roi_to_gasket` extends the ROI down to the dark holder
    # gasket, gated on finding that gasket so correctly-segmented cups are untouched.
    # It is geometrically CORRECT (verified: it reaches the gasket and re-includes
    # the cream mass on 50e294/749a, and is a no-op on cups already at their true
    # bottom). BUT enabling it does NOT fix the false-clean and REGRESSES the set:
    #   1. The lower cream mass is LARGER than the local-deviation base-blur kernel
    #      (dev_blur_kernel=121), so the masked blur adapts *to the cream* → its
    #      ΔE ≈ 0 and it is invisible to the detector regardless of ROI (confirmed
    #      even with the bright-neutral exclusion off). Detecting it needs a
    #      different sensor (global/region model or a smaller adaptive base), not a
    #      bigger ROI.
    #   2. Enlarging the ROI with the bright bottom band shifts the per-image
    #      adaptive threshold (mean+k·σ of ΔE), which flipped 18 cups and ERASED
    #      genuine detections elsewhere (incl. the cf4d chunk-extent fix).
    # Kept behind this flag (correct, reusable) for if/when the chunk detector is
    # made ROI-composition-robust. Set sam_bottom_extend_frac>0 to enable.
    sam_bottom_extend_frac: float = 0.0
    sam_gasket_dark_drop: float = 0.55

    # --- YOLO-seg SPILL detection (separate pipeline) ---
    # A trained "spill" seg model (training/train_multi.py --mode spill →
    # checkpoints/yolo_spill_seg.pt). Spill = any smoothie material OUTSIDE the
    # cup (drips on the gasket/holder, splatter on the machine). The SpillPipeline
    # unions every instance mask above spill_conf and reports the total spilled
    # area; a spill is DETECTED when that area is ≥ spill_min_area_px. The area
    # floor rejects tiny specks / single-instance noise that the model occasionally
    # fires on the gasket ring (the documented pale-on-steel / gasket-ring confound).
    spill_weights: Path = field(
        default_factory=lambda: Path("checkpoints/yolo_spill_seg.pt"))
    spill_conf: float = 0.35          # instance confidence floor for the mask union.
                                      # 0.35 not 0.25 (2026-07-15 disjoint eval): the
                                      # lowest-confidence detections (≈0.25–0.30) were the
                                      # spill FPs on the dark gasket edge / reflective steel;
                                      # every audited REAL spill scored ≥0.60, so 0.35 is a
                                      # safe precision floor. Genuine confident FPs on bright
                                      # specular steel (e.g. 225288 @0.89) survive — those are
                                      # a training-data gap (add steel-reflection negatives),
                                      # NOT reachable by a chroma gate (pale-on-steel confound:
                                      # real pale-smoothie spills measure chroma 7–10, BELOW
                                      # the steel FP's 11.7 — no separating threshold exists).
    spill_min_area_px: int = 400      # min total spill area (px) to call it a spill

    # --- YOLO-seg CHUNK detection (unblended-lump, trained-model PRIORITY) ---
    # A trained "chunk" seg model (training/train_multi.py --mode chunk ->
    # checkpoints/yolo_chunk_seg.pt), dispatched via smoothie_cv.detection.chunk.
    # Primary path: YOLO-standard ROI + YOLO-chunk masks; classical deviation
    # ensemble is the fallback when chunk weights are missing or inference fails.
    # chunk_yolo_input picks inference space (locked by
    # scripts/eval_chunk_yolo_input.py on labeling/chunk_dataset val+test):
    #   "full_filter" — full-frame YOLO, keep pixels inside smoothie ROI
    #   "roi_crop"    — crop to ROI, run YOLO on the crop
    chunk_detector_priority: list[str] = field(
        default_factory=lambda: ["yolo", "classical"])
    chunk_yolo_input: str = "full_filter"  # "full_filter" | "roi_crop"
    chunk_weights: Path = field(
        default_factory=lambda: Path("checkpoints/yolo_chunk_seg.pt"))
    chunk_conf: float = 0.25          # instance confidence floor for the mask union

    # --- output ---
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        if not _YAML_AVAILABLE:
            raise ImportError("pyyaml is required to load config from YAML. pip install pyyaml")
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})

    @classmethod
    def load(cls, yaml_path: str | Path | None = None) -> "Config":
        """Load from yaml_path if given, otherwise look for config.yaml in cwd."""
        default_path = Path("config.yaml")
        if yaml_path is None and default_path.exists():
            yaml_path = default_path
        if yaml_path is not None:
            return cls.from_yaml(yaml_path)
        return cls()
