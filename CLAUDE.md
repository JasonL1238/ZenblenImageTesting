# Project: ZenblenImageTesting — smoothie blendedness CV pipeline

# Commands
- Test (single):    `pytest smoothie_cv/tests/test_pipeline.py::TestClassicalCVPipeline -v`
- Test (full):      `pytest smoothie_cv/tests/test_pipeline.py -v`
- Run classical:    `python run.py --pipeline classical --image <img.jpg>`
- Batch:            `python run.py --pipeline classical --image data/images/ --threshold 0.90`

Classical is the only analysis pipeline; the VLM and SAM *analysis* pipelines
were removed. SAM2 remains the priority CONTAINER detector (see below).

# Container detection (ROI)
- SAM2 is the PRIORITY detector; classical colour-thresholding is the FALLBACK.
  SAM is colour-agnostic and robust across shades; classical is fragile on tan/pale.
- Single entry point: `from smoothie_cv.detection import detect_container`. It
  dispatches in `config.detector_priority` order (default `["sam", "classical"]`),
  falling back when a detector is unavailable or returns no plausible mask.
- Module layout under `smoothie_cv/detection/`:
  - `__init__.py`  → `detect_container()` dispatcher (SAM→classical) + public exports
  - `sam.py`       → `detect_sam()`        — SAM2 fixed-prompt detector [PRIORITY]
  - `classical.py` → `detect_classical()`  — colour-threshold + flatten [FALLBACK]
  - `common.py`    → shared helpers (classify, `flatten_roi_top`, `top_edge_roughness`, overlay)
- Force one: `detect_container(img, prefer="classical")`, or `run.py --detector sam|classical`
  (default `auto` = priority order).
- Compare methods head-to-head: `python scripts/compare_detectors.py [--sample]`.

# Unblended-chunk detection (inside ROI)
- After container detection, `ClassicalCVPipeline` finds unblended chunks inside
  the ROI. Two methods via `config.classical_method`:
  - `"deviation"` (DEFAULT) — colour-agnostic local-deviation: a chunk is a patch
    whose LAB colour deviates from the local base (masked large-kernel blur).
    Adaptive threshold (mean + k·σ), so it works on any smoothie shade.
  - `"canny"` — edge-boundary; only catches chunk RIMS, not bodies. Kept as a fallback.
- The "zenblen" LOGO is the systematic false positive — its colour/brightness/shape
  overlaps real chunks differently per smoothie shade, so NO single pixel threshold
  separates them. Handled by `_logo_text_labels()`: the logo is a horizontal ROW of
  similar-height marks spanning a wide extent; a chunk is a lone compact blob. Logo
  components are detected from the SAME ΔE map that fires the FP, then excluded.
- Chunks pass via THREE acceptance paths + foam-band cutoff + bright-neutral exclusion
  (cream logo/glare). Each colour-cued path carries a precision-preserving cue, so it
  can relax shape (`dev_relaxed_min_area`, `dev_relaxed_aspect_hi`) without flooding FPs:
  - compact path — area + solidity + extent + aspect + a strong deviation
    (`dev_compact_min_delta_e`). No colour cue, so precision rests on shape + ΔE.
  - dark path — relaxed shape but the component must be distinctly DARKER than the
    local base (`dev_dark_dL`). Recovers subtle chunks (a dissolving chunk's thin dark
    rim, e.g. yellow/62ed) that fail the compact gate. Darkness PRESERVES precision:
    logo/glare are brighter than base, so they never pass the dark path.
  - chroma path — relaxed shape but the component must be more SATURATED than the local
    base (`dev_chroma_dC`). Recovers HUE-similar chunks that are NOT darker — orange/amber
    lumps on a yellow smoothie (e.g. cf4d4c5, 99b2d39, 86240c8) deviate <16 ΔE and aren't
    dark, so compact+dark both miss them. Saturation PRESERVES precision the same way
    darkness does: glare/highlights/logo DESATURATE (ΔchromaC < 0), so they never pass.
  - Tuning note: the global ΔE floor was lowered 16→12 so hue-similar chunks form
    components at all; precision is then held by (a) compact path keeping the 16 floor,
    (b) the colour cues on dark/chroma, (c) foam cutoff raised 0.12→0.16 (the meniscus
    band just below the old cut fired the relaxed paths; real chunks sit at y_frac≥0.18).
    Net on the 92-image set: 21→27 flagged, recovering 6 real colour-similar chunks.
- After acceptance, each confident seed (≥ `dev_grow_min_seed_area`) is DIRECTIONALLY
  grown: the gates mask a chunk's high-contrast CORE, but its faint margin/tail (fading
  toward smoothie colour) deviates from base in the SAME colour direction, just weaker.
  Grow into contiguous pixels whose deviation projects ≥ `dev_grow_proj_thr` onto the
  seed's mean deviation direction (`_reconstruct`, distance-bounded by `dev_grow_max_iter`,
  reusing the glare/bright/foam exclusions). Completes the chunk (e.g. 4c68's lighter
  tail) without bleeding into smoothie/logo. Verdict-stable: still 27 flagged.
- PATH 4 (bottom absolute-chroma gate) — catches cream/pale masses at the cup BOTTOM
  that K=121 misses (large mass → local base adapts, ΔE≈0). Logic: if the MEDIAN
  absolute chroma of the last `dev_bot_n_rows` rows is ≤ `dev_bot_abs_chroma_max`=11,
  the bottom zone is flagged. Precision gates:
    - `dev_bot_min_body_L` ≥ 95: dark cups (maroon body L≈69) lose chroma at the
      hardware gasket and would false-positive — excluding them is mandatory.
    - `dev_bot_min_body_chroma` ≥ 22: skips pale/yellow bodies with no discriminative floor.
    - ABSOLUTE ceiling (≤11), NOT relative drop: the gasket transition on ANY cup
      dips 1–2 rows to ch≈5 but surrounding rows stay ≥12 → 6-row median stays above
      11 for clean cups. A cream mass fills ALL 6 rows uniformly → median ≈8–10.
      Drop-based logic fires on 38/92 images (natural gradient); absolute ceiling fires
      on 1 (50e294 with cream mass fully inside the ROI).
  Net on the 92-image set: 30→31 flagged (+1: 50e294 correctly recovered).
- PATH 5 (below-ROI cream-on-gasket band) — recovers thin cream layers that SAM cut
  ABOVE, so the cream sits just BELOW y_bot and Path 4 never sees it (e.g. 749a).
  Global gasket-extend was rejected first: enabling `sam_bottom_extend_frac` perturbs the
  per-ROI adaptive threshold and flips 20 unrelated borderline cups (10 gain / 10 lose,
  net 0) while recovering NONE of the targets — measured, see `[[bottom-cream-...]]`.
  Instead, scan the CENTRAL columns (`dev_botband_inset`) just below y_bot for a bright,
  slightly-warm low-chroma band bounded below by the dark gasket — the cream signature.
  When it fires, extend the ROI over that band and flag it; because the scan is gated on
  the signature, only genuine-cream cups change → ZERO churn (unlike global extend).
  Each look-alike is excluded by a DIFFERENT gate (robust across wide threshold ranges):
    - gray plastic holder clamp (db150e, ch≈0) → chroma floor `dev_botband_chroma_lo`=7
    - specular glare (L≈156)                   → L ceiling `dev_botband_L_hi`=145
    - dark gasket-edge shadow (L≈66–82)        → L floor   `dev_botband_L_lo`=100
    - chromatic smoothie                       → chroma ceiling `dev_botband_chroma_hi`=12
  Net on the 92-image set: 31→32 flagged (+1: 749a recovered). Verified surgically:
  Path-5-on vs off flips exactly 1 verdict (749a clean→chunks), 0 FPs, 0 churn.
  REMAINING LIMIT (db150e): the cup bottom is OCCLUDED by the gray holder clamp — there
  is no cream to recover and flagging the clamp would be a hardware FP. This is occlusion,
  NOT "cream below ROI" (the earlier diagnosis was wrong). Correctly stays clean.
  NOTE: when Path 5 fires, `BlendResult.mask` contains pixels BELOW the input ROI (the
  extended cream band); the "mask ⊆ input ROI" invariant becomes "mask ⊆ ROI ∪ cream band".
- KNOWN LIMIT: scattered tiny low-contrast flecks (dE≈8–12, <90px) on red/pink cups are
  NOT detected. They sit at the SAME local-contrast level as benign texture on truly
  well-blended cups (verified: a local black-tophat fires ~equally on clean 2e7754a and
  on chunky 4c68), so no global OR local threshold separates them without flooding FPs.
  This is a signal-floor limit, not a tuning gap — leave it unless a different sensor /
  per-region model is introduced.
- Validate across the dataset: `python scripts/validate_chunks.py` (uses cached SAM
  ROIs in `outputs/roi_cache/`; regenerate with `python scripts/cache_sam_rois.py`).
  Writes a clean report to `outputs/report/`: `flagged.png` (flagged smoothies,
  original vs detection side-by-side), `scores.csv`, `README.md`, `overlays/` (all 92).

# Code style
- Type-hint all public function signatures
- No hardcoded API keys — always read from environment variables

# Workflow
- Prefer running a single test class over the full suite for speed.
- After CV edits: write result image to `outputs/`, then READ it back — never assert
  success without inspecting the actual mask overlay.
- Gate "done" on a numeric blend_score in [0, 1], not on the image looking right.
- Use a subagent for image analysis and multi-file investigation to keep context lean.

# Gotchas / environment
- SAM2 container detection requires checkpoint files in `checkpoints/` — download separately:
  `wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P checkpoints/`
- SAM2 is installed from source, not PyPI: see `requirements.txt` comments.
- M4 Pro uses MPS backend (`torch.backends.mps`); Jetson Nano falls back to CPU — no CUDA.
- Pipelines implement the `BlendResult` contract (`smoothie_cv/pipelines/base.py`).

# Session health (canary)
- Begin EVERY response with the marker 🟢 followed by a space.
- This is a context-health check — never skip it. If it starts disappearing,
  the session context is degrading: /clear (or /compact) and re-anchor.

# Compaction
- When compacting, always preserve: the list of modified files, the chosen
  approach and WHY, and any test / run commands.
