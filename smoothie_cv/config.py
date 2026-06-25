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
    # relaxed shape gates shared by the dark + chroma paths (both already carry a
    # colour/darkness cue that logo text lacks, so shape can be looser than compact).
    dev_relaxed_min_area: int = 90    # min blob area for the colour-cued paths (< dev_min_area)
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
    dev_grow_min_seed_area: int = 200   # only GROW seeds at least this big (a confident
                                        # chunk core worth completing). Tiny marginal seeds
                                        # (logo letter, rim glare, lone fleck) are kept as-is
                                        # but NOT amplified — growth must not flip a near-clean
                                        # smoothie to flagged on a borderline speck.
    dev_glare_L: float = 240.0     # LAB L* above this + low chroma = specular glare
    dev_glare_chroma: float = 12.0
    # printed-logo / backlit-text exclusion: bright-vs-base + low chroma (cream text)
    dev_bright_dL: float = 25.0    # ΔL above local base to be considered "bright text"
    dev_bright_chroma: float = 22.0
    # logo text-line detector: a row of similar-height marks spanning a wide extent
    dev_letter_min_area: int = 60      # min component area to be a letter candidate
    dev_letter_h_lo: float = 0.025     # letter height range as fraction of ROI height
    dev_letter_h_hi: float = 0.20
    dev_text_min_letters: int = 3      # min aligned letters to call it a text line
    dev_text_min_span: float = 0.32    # min horizontal extent of the word (frac ROI width)
    dev_text_height_cv: float = 0.35   # max letter-height coeff. of variation (uniformity)

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
    # Order detectors are tried in. SAM2 is the priority detector (colour-agnostic,
    # robust across shades); classical colour-thresholding is the fallback, used
    # only when SAM is unavailable or returns no plausible mask.
    detector_priority: list[str] = field(default_factory=lambda: ["sam", "classical"])

    # --- SAM2 (container detection) ---
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
