# Project: ZenblenImageTesting — smoothie blendedness CV pipeline

# Commands
- Test (single):    `pytest smoothie_cv/tests/test_pipeline.py::TestClassicalCVPipeline -v`
- Test (full):      `pytest smoothie_cv/tests/test_pipeline.py -v`
- Run classical:    `python run.py --pipeline classical --image <img.jpg>`
- Batch:            `python run.py --pipeline classical --image data/images/ --threshold 0.90`

Classical is the only analysis pipeline; the VLM and SAM *analysis* pipelines
were removed. A fine-tuned YOLO-seg model is the priority CONTAINER detector;
SAM2 is phased out of the pipeline but kept as the training-free reference.

# Container detection (ROI)
- The fine-tuned YOLO11n-seg model is the PRIORITY detector (trained on our own
  smoothie-only labels — foam excluded, reaches the true cup bottom, 6 MB nano
  net). Classical colour-thresholding is the FALLBACK. SAM2 is LEGACY: kept in
  the codebase (needs no training data → independent reference for evaluating
  newly trained models, and the labeling tool's bootstrap segmenter) but NOT in
  the default priority.
- Single entry point: `from smoothie_cv.detection import detect_container`. It
  dispatches in `config.detector_priority` order (default `["yolo", "classical"]`),
  falling back when a detector is unavailable or returns no plausible mask.
- Module layout under `smoothie_cv/detection/`:
  - `__init__.py`  → `detect_container()` dispatcher (YOLO→classical) + public exports
  - `yolo.py`      → `detect_yolo()`       — fine-tuned YOLO-seg [PRIORITY]; also
                     `get_yolo_roi()` + `fill_holes()` used by the scripts
  - `classical.py` → `detect_classical()`  — colour-threshold + flatten [FALLBACK]
  - `sam.py`       → `detect_sam()`        — SAM2 fixed-prompt detector [LEGACY]
  - `common.py`    → shared helpers (classify, `flatten_roi_top`, `top_edge_roughness`, overlay)
- Force one: `detect_container(img, prefer="sam")`, or `run.py --detector yolo|sam|classical`
  (default `auto` = priority order).
- Deployed weights: `checkpoints/yolo_smoothie_seg.pt` (`config.yolo_weights`).
  Train → deploy loop (see train.py docstring):
  1. label in `labeling/`, export via `python dataset_pipeline.py export-all`
  2. `/opt/miniconda3/bin/python train.py` (bump RUN_NAME first)
  3. sanity-check masks vs the SAM reference:
     `python scripts/compare_yolo_vs_sam.py --weights runs/smoothie-seg/<run>/weights/best.pt`
  4. `cp runs/smoothie-seg/<run>/weights/best.pt checkpoints/yolo_smoothie_seg.pt`
  5. `rm -rf outputs/roi_cache_yolo && python scripts/cache_yolo_rois.py`
  6. `python scripts/validate_chunks.py` — MUST re-validate all 92 (ROI changes
     shift the adaptive chunk threshold and flip verdicts).

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
- CONFIRMED-WORDMARK BAND SUPPRESSION (`dev_logo_band_suppress`, added 2026-07 from
  the 500-image labeling-disjoint eval): once `_logo_text_labels()` CONFIRMS a
  wordmark, reject any OTHER accepted component whose centroid lands inside that
  wordmark's bounding band (± `dev_logo_band_margin_frac`·median-letter-height) AND
  whose area ≤ `dev_logo_band_max_area_mult`·median-letter-area. Fixes the leak where
  a counter-shadow blob or a stray letter that did NOT join the group rides alongside
  the confirmed letters — colour-identical to a chunk (dark+saturated), so ONLY size +
  position-in-a-confirmed-band separate them. The area ceiling is mandatory: a real
  chunk overlapping the wordmark is a distinctly larger mass (measured: leaked ≤274px
  vs real chunks 875/1752px in the same band — both real chunks kept). Conditional on a
  confirmed wordmark → never fires on cups with no detected logo. Net: 92-set 32→31
  (removed the c6f69e01 counter-shadow FP — a previously-uncaught logo FP, NOT a lost
  chunk); 500-set eval −7 logo FPs, 0 real chunks lost, 0 new flags. LIMIT: does NOT
  fix the CLIPPED-wordmark case (word partly out of frame / curved on the cup → <3
  letters or span <`dev_text_min_span`, so the text-line detector never fires and there
  is no confirmed band). That residual (~13 images on the 500-set) needs a trained YOLO
  logo-mask class; measured that NO count/span relaxation catches them without also
  grouping real 2-chunk clusters (the documented "no threshold separates them" wall).
- TWO-PASS BASE: bright print pulls the pass-1 K=121 base UP around the letters, so
  ordinary smoothie between/inside them reads "darker than base" — the print's
  COUNTER-SHADOW (dark-path FP, e.g. the blob in the "n" counter on 28fe/560d).
  Pass 2 re-estimates the base EXCLUDING print-signature pixels (ΔL>+25 + absolute
  chroma<22) from the masked blur, like outside-ROI pixels. Root-cause fix: the
  counter-shadow never forms, and letters contrast harder (text-line detector sees
  them more reliably). Bottom zone keeps pass-1 behaviour (cream exemption intact).
- COMPONENT-LEVEL print/glare gates (added when YOLO ROIs shifted the adaptive
  threshold and let single-letter fragments through; all keyed on colour RELATIVE
  to base or ROI geometry — no absolute colours, hold across shades):
  - bright-desat: mean ΔL > +25 AND mean ΔC < 0 above the bottom exempt zone =
    print/glare, reject. Kills lone CLIPPED letters ("ze…" at the frame edge,
    e89d/c7d2) that defeat the ≥3-letter text-line detector and ride above the
    absolute-chroma pixel rule on saturated smoothies.
  - chroma-path brightness ceiling `dev_chroma_dL_max`=+5: on PALE/tan cups warm
    print is MORE saturated than the body (ΔC≈+8 — defeats the "logo desaturates"
    assumption) but backlit-bright (ΔL +11…+14); real saturated lumps measure
    ΔL −8…+1 (54121aaf/dbc7 killed; cf4d/99b2/86240 untouched).
  - dark-path position gate `dev_relaxed_top_frac`=0.18 + print-halo overlap
    (belt-and-suspenders for the counter-shadow): meniscus shadow hangs just below
    the foam cut, real dark chunks sit at y_frac ≥ 0.29 (abef9780 killed).
  - chroma-path meniscus interiority `dev_chroma_band_interior`=0.7: in the band
    above 0.18, rim/wall-junction slivers HUG the ROI contour (measured 0.28
    interior at 2×erode) while a real surface lump is fully interior (1.00,
    f0b6a6d1 kept, 5149243f killed).
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
- PATH 6 (reference-band pass, `dev_global_enable`, re-enabled 2026-07): with ROIs
  that reach the true cup bottom (YOLO-seg), large cream masses sit fully INSIDE the
  ROI where K=121 adapts to them (paths 1–3 blind) and Path 4's last-rows gate can
  sit on gasket pixels and miss. This pass compares the lower zone against a mid-cup
  reference band (ΔE≥22 + a colour-jump gate, additive — does NOT perturb the
  adaptive threshold). The colour-jump gate has TWO branches (union — one physical
  mass often has both):
    - NEUTRAL (chroma-drop≥12): cream/neutral masses desaturate vs the body.
    - HUE (`dev_global_hue_deg`=35° + chroma≥12 floor): a saturated mass of a
      different colour FAMILY (yellow banana on red, 50e294) keeps chroma similar
      to the body — the neutral branch is BLIND to it (50e294's warm part measured
      drop≈13, barely over the gate; Path 4 saw bot chroma 18.1 > its ≤11 cream
      ceiling) — but swings the LAB ab-plane hue angle. Natural gradients lighten
      within the body's hue family. Margin measured on the 92-set: only genuine
      bottom masses (50e294, 749a) produce a bottom-attached hue component at ANY
      θ in 25–50°; clean cups produce zero. The chroma floor excludes near-neutral
      pixels whose hue is numerically unstable (glare/shadow).
  Precision gates, each killing a distinct look-alike:
    - area ≥ 1500px (logo letters, glare specks)
    - BOTTOM ATTACHMENT `dev_global_bot_attach_frac`: cream is heavy and RESTS on
      the gasket; the diffuse backlit condensation glow floats mid-low cup detached
      from the last rows — without this gate the glow flags ~6 clean cups
      (3fe5f4c7, f0688700, e0ef3dcf, …).
    - accepted components are geodesically reconstructed into the UN-eroded ROI
      (dE_ref has no masked-blur boundary artifact) with an L≥100 floor so the mass
      fills to the cup wall/gasket without grabbing the dark gasket edge.
      Reconstruction uses its OWN bound `dev_global_grow_max_iter`=512, NOT the
      shared dev_grow_max_iter=40 (that protects paths 1–3's looser field; a bottom
      mass spans ~250px and 40px of reach left 2/3 of 50e294 uncovered). Safe by
      construction: clean cups have no ≥1500px bottom-attached seed, so nothing grows.
  Result: 749a/50e294 masses fully covered wall-to-wall under YOLO ROIs (50e294
  3854→6172px incl. the warm-yellow lobe, 749a 5716→7738px); verdicts unchanged
  (32 flagged, 0 flips).
- YOLO ROI masks are HOLE-FILLED at the source (`get_yolo_roi`): a liquid mask is
  simply connected; the model punching holes over logo letters breaks the text-line
  logo exclusion (the word is no longer a row of marks inside the ROI → 28fe82f8).
- `dev_grow_min_seed_area` 200→100: with the component gates above, surviving seeds
  are clean enough to grow; recovers small real chunks that stalled sub-flag under
  YOLO ROI threshold shifts (afdc6c3e 102→547px, 8343d981 112→625px).
- KNOWN LIMIT: scattered tiny low-contrast flecks (dE≈8–12, <90px) on red/pink cups are
  NOT detected. They sit at the SAME local-contrast level as benign texture on truly
  well-blended cups (verified: a local black-tophat fires ~equally on clean 2e7754a and
  on chunky 4c68), so no global OR local threshold separates them without flooding FPs.
  This is a signal-floor limit, not a tuning gap — leave it unless a different sensor /
  per-region model is introduced.
  SAME FLOOR, CHROMA PLANE (measured 2026-07): soft hue-similar lumps on pale yellow
  (636e83f4, dC +8.6…+9.7, total ΔE < 12) are inseparable from the embossed-ridge /
  glow-rim strips on audited-clean tan cups (dC +7.7…+9.6) — a chroma-plane deviation
  path (Path 7, `dev_chroma_plane_enable`, kept DISABLED) recovers 1 cup and flips 4
  clean ones. Zero margin; do not tune, fix optics. Also still missed: 0ad03494
  (documented mid-bottom cream, interleaved with clean cups) and dd4d2902 (frame
  unanalyzable: cup behind frost/condensation — capture QA issue, both ROIs poor).
- Validate across the dataset: `python scripts/validate_chunks.py` (uses cached
  YOLO ROIs in `outputs/roi_cache_yolo/`; regenerate with
  `python scripts/cache_yolo_rois.py`; `--live` runs the model instead; SAM
  reference cache: `outputs/roi_cache_sam/` via `scripts/cache_sam_rois.py`).
  Writes a clean report to `outputs/report/`: `flagged.png` (flagged smoothies,
  original vs detection side-by-side), `scores.csv`, `README.md`, `overlays/`
  (all 92) — and diffs verdicts vs the previous report (`--baseline`), printing
  new/lost flags. Compare any two runs: `scripts/diff_reports.py a b`.
  Per-image gate tracing: `scripts/debug_chunk_paths.py <stem> [--roi yolo|sam|both]`
  (MUST be kept in sync with `_deviation_mask` — it replicates the logic).

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
- YOLO container detection needs `checkpoints/yolo_smoothie_seg.pt` (gitignored —
  retrain with `train.py` and promote best.pt, or copy from another machine).
- SAM2 (legacy detector) requires checkpoint files in `checkpoints/` — download separately:
  `wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt -P checkpoints/`
- SAM2 is installed from source, not PyPI: see `requirements.txt` comments.
- M4 Pro uses MPS backend (`torch.backends.mps`); Jetson Nano falls back to CPU — no CUDA.
  YOLO-seg TRAINING segfaults on MPS — train.py forces CPU.
- Pipelines implement the `BlendResult` contract (`smoothie_cv/pipelines/base.py`).

# Session health (canary)
- Begin EVERY response with the marker 🟢 followed by a space.
- This is a context-health check — never skip it. If it starts disappearing,
  the session context is degrading: /clear (or /compact) and re-anchor.

# Compaction
- When compacting, always preserve: the list of modified files, the chosen
  approach and WHY, and any test / run commands.
