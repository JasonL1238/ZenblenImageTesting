# Chunk-segmentation ablation — report

**Isolated evaluation of new chunk-detection methods inside an already-good
smoothie mask. Nothing here is integrated into production.**

---

## 1. Executive summary

Seven candidate methods for detecting unblended chunks *inside* the existing
smoothie ROI were implemented and run on the same 60-image test set (10 hard/
failure cases + 50 previously-clean regression cups), reusing the production
YOLO smoothie masks (`outputs/roi_cache_yolo/`). No method re-segments the
smoothie; all use adaptive per-image percentile thresholds inside the ROI.

**Bottom line: none of the seven is safe to drop in as a replacement for the
production chunk detector, and none should be integrated standalone.** The
reason is precision, not sensitivity:

- Production's colour-deviation chunk detector has **~perfect precision** on
  this data — a visual audit of all 26 production `chunks` verdicts found **0
  logo/glare/hardware false positives**. Its only weakness is *recall* on
  documented signal-floor misses.
- Every low-level cue tested here (residual, DoG, texture, gradient) fires on
  the **"zenblen" wordmark**, which is locally dark/edge-dense/compact — exactly
  like a chunk. Production solves this with a *dedicated* logo suppressor
  (`_logo_text_labels()` + wordmark-band suppression); these raw methods have no
  such stage.
- The **ensemble** (cross-cue agreement gate) is the best of the seven — it
  shrinks the wordmark response from the whole word down to a single glyph — but
  a visual audit still found a **logo false positive (usually the letter "e") on
  ~21 of 50 clean cups (~40%)**, versus production's 0%. That is a precision
  **regression**.

**What IS worth taking forward** is the *primitive*, not any method as-is: the
**colour-deviation residual + a cross-cue agreement gate**, layered **on top of
production's existing logo mask**, is a credible way to add recall on the
signal-floor misses (it recovered `4c680c8`, `749a1809` and a plausible lump on
`e54d4dcf` in this run) without re-introducing logo FPs. See §8.

---

## 2. Test images

60 images, drawn from the 92-image production set (`data/images/{red_pink,yellow}`).
Smoothie ROI reused from `outputs/roi_cache_yolo/<stem>.png`. Full lists in
`test_sets.py`.

No pixel-level ground truth exists, so evaluation is **comparison-based**:
production verdict (`outputs/report/scores.csv`) + **visual overlay inspection**
+ candidate-count sanity checks. All test-set membership and all failure/regression
judgements below come from actually looking at the overlays (see the montages in
`results/_audit_figures/` and the full `gallery.html`).

### 2A. Bad / failure set (10) — where production struggles to detect chunks
Selected from the documented KNOWN LIMITS in `CLAUDE.md`/memory and confirmed by
inspecting each production overlay.

| stem (8-char) | category | why production struggles |
|---|---|---|
| 636e83f4 | MISS | soft hue-similar lumps on pale yellow (ΔC<10); verdict clean |
| 0ad03494 | MISS | mid-bottom cream, interleaved with clean cups; verdict clean |
| e54d4dcf | MISS | 2–3 discrete dark specks in a smooth yellow body; verdict clean |
| 09291f76 | SIGNAL-FLOOR | scattered tiny flecks, only barely flagged (0.9945) |
| 4c680c8 | FAINT-TAIL | maroon chunk with a fading tail; core caught, tail not |
| 749a1809 | BOTTOM-CREAM | cream band low in cup; hard to localize (worst-scored flag) |
| 50e294fa | BOTTOM-MASS | yellow banana lobe on red body; hue-shift case |
| dd4d2902 | UNANALYZABLE | cup behind frost/condensation; both ROIs poor |
| db150ed5 | CONTROL-OCCLUSION | gray holder clamp occludes cup bottom; correctly clean |
| 2e7754a2 | CONTROL-CLEAN | benign red/pink texture; must stay clean |

### 2B. Good / regression-protection set (50) — must-not-break
50 production-`clean` cups, shade-balanced (36 red_pink + 14 yellow), each visually
confirmed well-blended (no undetected chunk) across three inspection passes. Any
detection a new method makes here is a potential false positive.

---

## 3. Methods and parameters

All under `methods/<name>.py`, satisfying a common contract
(`NAME`, `DEFAULT_PARAMS`, `detect(img_bgr, roi_mask, params) -> MethodResult`),
using only `common` helpers, LAB colour space, ROI erosion to exclude the rim,
and **adaptive per-image percentile thresholds** (no fixed raw cutoffs).

| Method | Idea | Key params (defaults) |
|---|---|---|
| `residual_highpass` | \|channel − large-blur background\|, LAB L+ab terms | bg_k=121, pct=98, erode_frac=0.05, min_area=120, min_solidity=0.55 |
| `dog_blob` | multi-scale \|DoG\| over σ=[2,4,8,16], L+chroma | pct=98.5, min_area=120, min_circularity=0.15 |
| `texture_variance` | local variance boxfilter(I²)−boxfilter(I)² on L | win_k=9, post_blur_k=11, pct=98, min_area=120 |
| `gradient_magnitude` | Scharr + 0.5·\|Laplacian\|, zeroed-then-blurred | density_blur_k=21, pct=98.5, min_area=150 |
| `morphology_contour` | residual candidate + aggressive morphology/shape cleanup | open_k=5, close_k=7, min_solidity=0.55, extra edge erode |
| `watershed_split` | candidate → EDT → marker watershed to split touching blobs | peak_dist_frac=0.06, smooth_dist_sigma=3.0 |
| `ensemble` | **score = residual; require all 3 cues to agree (gate)** | mode=weighted_sum(w_res=1), min_agree=3, residual_required, pct=98 |

Full per-image params are recorded in every `results/<method>/json/<stem>.json`.

---

## 4. Results table

Per-method aggregates over the shared test set (raw source:
`results_summary.csv`, one row per image per method). "clean n=0" = clean cups
with zero detections (higher = better precision); "good total chunks" = total
candidates on the 50 clean cups (lower = fewer FPs).

| method | good mean n | good imgs flagged | good total chunks | good clean (n=0) | bad w/ detection | bad mean n | ms/img |
|---|---|---|---|---|---|---|---|
| residual_highpass | 1.90 | 45/50 | 95 | 5 | 9/10 | 1.6 | 9 |
| dog_blob | 1.46 | 39/50 | 73 | 11 | 9/10 | 1.7 | 117 |
| texture_variance | 2.84 | 50/50 | 142 | 0 | 10/10 | 3.7 | 5 |
| gradient_magnitude | 2.14 | 50/50 | 107 | 0 | 10/10 | 2.5 | 6 |
| morphology_contour | 2.94 | 50/50 | 147 | 0 | 10/10 | 3.5 | 13 |
| watershed_split | 5.88 | 50/50 | 294 | 0 | 10/10 | 6.6 | 21 |
| **ensemble** | **1.62** | **42/50** | **81** | **8** | 9/10 | 1.6 | 29 |

**Reading it:** raw counts rank `ensemble` and `dog_blob` best on precision and
`watershed_split` worst (it over-segments — it is a splitter, not a detector).
But counts alone are misleading, because a detection that lands on the logo
"counts" the same as one on a chunk. The visual audit (§6) is what actually
decides precision.

---

## 5. Failure-case (bad-set) recovery analysis — ensemble

Per-stem, from inspecting `results/ensemble/overlay_in_roi/` vs the production
overlay (montage: `results/_audit_figures/ens_bad.png`):

| stem | outcome | detail |
|---|---|---|
| 4c680c8 | **RECOVERED** | contour lands tightly on the brown chunk |
| 749a1809 | **RECOVERED** | contours on the bottom cream mass (partial coverage) |
| e54d4dcf | **RECOVERED (new recall)** | plausible lump low-centre; production missed it |
| 636e83f4 | PARTIAL | catches some soft yellow lumps low in the cup |
| 50e294fa | PARTIAL | catches part of the bottom mass **+ a logo "e" FP** |
| 0ad03494 | **NOT recovered** | misfires on the logo "e", not the mid-bottom cream |
| 09291f76 | **NOT recovered** | flecks stay below floor; fires on the logo instead |
| dd4d2902 | n/a | unanalyzable (frost); detection meaningless |
| db150ed5 | **FALSE POSITIVE** | logo "e" flagged; this occlusion cup should stay clean |
| 2e7754a2 | **CLEAN (correct)** | no detection; wordmark not flagged |

So the ensemble genuinely **adds recall on 3 documented misses** (`4c680c8`,
`749a1809`, `e54d4dcf`) and partials two more — but it does **not** solve the
hardest signal-floor cases (`09291f76`, `0ad03494`), and on those it tends to
mis-spend its detection on the logo.

---

## 6. Regression analysis on the 50 good cups — the deciding result

Visual audit of `results/ensemble/overlay_in_roi/` for all 50 clean cups
(montages `results/_audit_figures/ens_good_1.png`, `ens_good_2.png`):

| bucket | count / 50 | meaning |
|---|---|---|
| fully clean (n=0) | ~8 | ideal |
| **LOGO false positive** | **~21** | red contour on a wordmark glyph (usually "e") — **regression** |
| benign-texture / bottom-lump / glare | ~21 | borderline; small, some plausible, some FP on bubbles/glare |

**The wordmark "e" is the dominant, systematic false positive.** The cross-cue
agreement gate suppresses the *full* wordmark (which `texture_variance` lights
entirely) down to a single compact glyph, but the "e"/"z"/"b" closed-loop letters
are dark + edge-dense + compact + solid — they pass all three cues and the shape
filter. This is the *same* failure mode the CLAUDE.md docs describe ("no single
pixel threshold separates the logo from chunks"), and it is exactly why
production needs a *dedicated* logo detector.

For comparison, the other methods are worse here: `texture_variance` /
`morphology_contour` light the whole wordmark on nearly every clean cup;
`gradient_magnitude` collapses it to one or two letter-blobs; `dog_blob` leaks
letter strokes and glare; `residual_highpass` is the cleanest single cue
(n=0 on the 2e7754a control) but still ~45/50 flagged.

**Net regression verdict:** every method, ensemble included, would *reduce*
precision vs production (0 logo FPs → ~40% logo-FP rate for the best method) if
used on its own.

---

## 7. Per-dimension scoring (all methods)

| dimension | residual | dog | texture | gradient | morph | watershed | ensemble |
|---|---|---|---|---|---|---|---|
| False-positive risk (logo/glare) | high | high | **very high** | high | **very high** | **very high** | **medium** |
| Missed-chunk risk | medium | medium | low | low | low | low | medium |
| Oversegments texture/bubbles | med | med | high | med | high | **very high** | med |
| Misses low-contrast chunks | some | some | few | few | few | few | some |
| Detections near rim/edge | low* | low* | low* | low* | **lowest** | low* | low* |
| Runtime / image | 9ms | 117ms | 5ms | 6ms | 13ms | 21ms | 29ms |
| # params needing tuning | ~6 | ~7 | ~7 | ~8 | ~10 | ~9 | ~7 |
| Generalizability (shade-robust) | good | good | good | good | good | good | good |

\* rim FPs are already controlled by `roi_interior` erosion in `common.py`.

All methods are colour-agnostic (LAB + percentile) and generalize across red/pink
and yellow shades — none needs per-shade constants. Parameter sensitivity is
moderate: the ensemble agent measured that `min_agree=2`, any nonzero
texture/gradient *score* weight, `pct=99`, or `min_solidity=0.60` each either
re-introduce a wordmark FP or drop a real mass — i.e. the good operating point is
a fairly narrow ridge.

---

## 8. Recommendation

**Do not integrate any of these methods standalone.** Recommended path, in order
of value:

1. **Best single method for the stated goal:** `ensemble` (residual score + 3-cue
   agreement gate). It has the best precision of the seven and recovers 3
   documented misses. But it is only safe **if gated by production's existing
   logo mask** — see step 2.
2. **The actually-integratable idea:** add the ensemble as an *additive recall
   pass* inside the production deviation pipeline, **after** `_logo_text_labels()`
   has produced the wordmark mask, and **subtract that logo mask** from the
   ensemble's candidates before accepting them. This directly removes the ~40%
   logo-FP regression (the only thing blocking it) while keeping the recovered
   misses. It mirrors how production's existing extra paths (Path 4–6) are
   *additive and signature-gated* rather than threshold changes.
3. **`residual_highpass`** is the cleanest primitive if you want the simplest
   possible add (single cue, 9 ms, n=0 on the control); the ensemble's agreement
   gate mainly helps by trimming benign texture, less so the logo.
4. `watershed_split` is only useful as a **post-step** to split a confirmed merged
   mass into instances (it does that correctly and does not shatter single
   chunks); it is not a detector.
5. `texture_variance`, `gradient_magnitude`, `morphology_contour`, `dog_blob`:
   keep as ablation references only — they do not beat the residual on precision.

### Is it safe to integrate? 
Not as-is. **Safe only after logo-masking (step 2).** Before that, expect a large
precision regression on clean cups.

### What it fixed vs didn't
- **Fixed / recovered:** `4c680c8` (faint-tail chunk), `749a1809` (bottom cream),
  `e54d4dcf` (dark-speck miss, brand-new recall), partial `636e83f4`/`50e294fa`.
- **Did NOT fix:** `09291f76` and `0ad03494` (true signal-floor — inseparable from
  benign texture, matching the documented wall), `dd4d2902` (unanalyzable capture).

---

## 9. Next implementation steps (only if you decide to integrate)

These are the files that *would* be touched later — **none touched now**:

1. `smoothie_cv/pipelines/classical_cv.py` — where `_deviation_mask` /
   `_logo_text_labels` run. Add an optional additive "agreement recall" pass that:
   consumes the same ΔE residual it already computes, requires texture+gradient
   agreement, and **excludes the logo component mask** already produced there.
   Port the logic from `methods/ensemble.py` (agreement gate) + `residual_highpass.py`.
2. `smoothie_cv/config.py` — add gated knobs (`dev_agree_enable`, `min_agree`,
   percentiles) defaulting **off**, so production behaviour is unchanged until
   explicitly enabled.
3. `scripts/validate_chunks.py` — re-validate all 92 (per CLAUDE.md, any change to
   the chunk stage shifts the adaptive threshold and can flip borderline verdicts).
   Diff against the current report baseline; require **0 new logo FPs** and count
   recovered misses before promoting.
4. `scripts/debug_chunk_paths.py` — mirror the new path so per-image tracing stays
   in sync (as the docs require).

No change to container detection, ROI caching, or training is implied.

---

## 10. Risks and limitations

- **No pixel ground truth.** All precision/recall judgements are visual +
  count-based, on 60 of 92 images. Numbers are directional, not exact.
- **The logo FP is fundamental to these cues** — this experiment re-confirms the
  documented "no pixel threshold separates logo from chunk" wall; the only fix is
  the *dedicated* logo mask, not tuning.
- **Signal-floor misses remain unsolved** (`09291f76`, `0ad03494`) — consistent
  with the documented conclusion that these need better optics, not software.
- **Narrow operating point** for the ensemble (see §7); worth re-tuning against a
  full-92 sweep before trusting it in production.
- **Threshold-shift risk:** integrating anything into the deviation stage perturbs
  the adaptive threshold; the mandatory 92-image re-validation (step 3) is how you
  catch flips.
- The ensemble runs the 3 base cues per image (~29 ms) — fine on the M4, verify on
  the Jetson (CPU) before deploying.

---

*Reproduce: see `README.md`. Visual comparison: open `gallery.html`; audit
montages in `results/_audit_figures/`.*
