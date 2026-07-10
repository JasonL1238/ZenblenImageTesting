# Feasibility Test #Y — Trained YOLO "logo" class to suppress residual logo FPs

Date: 2026-07-06 · Scope: isolated proof-of-concept. Nothing in smoothie_cv/,
config.py, checkpoints/, or outputs/report* was touched. All artifacts live under
experiments/logo_clipped_fix/yolo_test/.

## Verdict
FEASIBLE. A rough bootstrap detector — trained on auto-generated labels, in a few
minutes on CPU — already WOULD SUPPRESS 5 of the 7 residual logo false-positives.
Because the labels are weak and partial (see caveat), treat 5/7 as a FLOOR, not a
ceiling. A properly hand-labeled model should do better, specifically on the one miss
that is genuinely a clipped wordmark.

## What was built
- Dataset (dataset/, data.yaml): 37 positive + 18 negative images, 1 class `logo`.
  Labels BOOTSTRAPPED from the OpenCV EAST scene-text detector confined to the ROI (on
  these cups the only in-ROI text is the zenblen wordmark), merging collinear letter
  boxes into a wordmark bbox. Positives = high logo_suspicion cups (manifest) with a
  clean EAST cluster + the 7 confirmed frontal shorts; negatives = low-suspicion cups
  with no EAST cluster. All 7 residual targets held OUT of train/val.
- Model (runs/logo-class-test/): yolo11n.pt detection (bbox is enough; seg not needed),
  40 epochs, imgsz 640, device=cpu (seg segfaults on MPS; detection on CPU ~7 min).
  Bootstrap-val: mAP50 0.995, P 0.96, R 1.00.
- Eval (evaluate.py, results.csv, overlays/): predict logo boxes on each held-out
  residual; "would-suppress" if a predicted box overlaps the actual leaked chunk
  fragment. Leak location obtained by running the REAL pipeline READ-ONLY.

## Results on the 7 held-out residuals
| short    | leaked fragment          | logo box fired    | overlaps leak | outcome |
|----------|--------------------------|-------------------|---------------|---------|
| 4bf0d44c | mid-cup                  | yes (0.60)        | yes           | SUPPRESS |
| 8572bc76 | mid-cup                  | yes (0.88)        | yes           | SUPPRESS |
| a3dce37a | clipped "ze"             | yes (0.41)        | yes           | SUPPRESS |
| d71aa159 | mid-cup                  | yes (0.87)        | yes           | SUPPRESS |
| adfcd90b | mid-cup                  | yes (0.56)        | yes           | SUPPRESS |
| dce9974f | L-edge clipped wordmark  | NO (0.0)          | no            | MISS (clipped) |
| 572b0f91 | cup BOTTOM               | yes at top (0.84) | NO            | MISS (leak not at wordmark) |

5/7 would be suppressed. 6/7 got any logo detection.

### The two misses are different in kind
- dce9974f — the real clipped-wordmark case. Faint, left-edge-clipped/curved wordmark on
  a pale cup. The bootstrap model never fired. This is the INHERITED EAST blind spot:
  EAST returns nothing on clipped upright text, so no clipped-wordmark examples entered
  the training labels. A hand-labeled model that deliberately includes clipped/curved
  wordmarks is the fix — exactly the case the logo-class idea was motivated by.
- 572b0f91 — NOT a wordmark leak at all. The model correctly detects the wordmark at the
  top of the cup, but the leaked fragment sits at the cup BOTTOM (y~474-535, near the
  gasket / a "perfectly blended" tagline), nowhere near the main wordmark. A logo class
  cannot suppress this by overlap; it needs its own gate. No amount of wordmark labeling
  fixes it.

### Generalization signal (why 5/7 is a floor)
On a3dce37a, EAST's box clustering returned nothing (the bootstrap pipeline would not
have labeled it), yet the TRAINED YOLO fired on the clipped "ze" and overlapped the leak.
The model learned the wordmark appearance and generalized BEYOND its own weak labels —
direct evidence that better labels raise the ceiling.

## False-fire on clean cups — and why the headline number is misleading
33/40 sampled clean cups fired a logo box (high conf, 0.9+). This is NOT a defect. Those
cups have a visible zenblen wordmark, so a logo detection there is correct; it is harmless
because a clean cup has no real chunk to erase. (I sampled cups with manifest
logo_suspicion <= 0.4; that heuristic under-counts visible logos — the trained model
finds wordmarks it missed.) The metric that actually matters for production is UNTESTED
here: does a logo box ever overlap a REAL chunk on a chunky cup and erase it? That needs a
known-chunky fixture set. This is the same risk the current dev_logo_band_suppress already
guards with an area ceiling — so a logo class should be used as a GATED signal (suppress
only letter-sized components inside the box, keep the larger-mass exemption), never as a
blanket exclude.

## Bootstrap-label-quality caveat
Labels came from EAST, not a human. They are noisy and often PARTIAL — several boxes cover
only part of the word (e.g. "blen", "zenble") and a few are small/off. The model learned
from imperfect targets, which under-states what a clean-labeled model achieves. Frame 5/7
as the FLOOR. EAST also structurally cannot label the clipped-wordmark case (dce9974f),
which is precisely the residual we most want to catch.

## Label-cost estimate for a production-grade model
- Approach: correct, don't create. Pre-label all ~500 corpus images with the EAST
  bootstrap boxes (already scripted here), then hand-correct in the existing labeling/
  tool. Correcting a pre-drawn box is far faster than drawing from scratch.
- Volume: ~150-250 hand-verified images give good shade + geometry coverage. MUST
  explicitly include the clipped/curved/edge-cropped wordmark cases (~13 on the 500-set
  per CLAUDE.md) and the frontal wordmark across all shades, plus negatives.
- Effort: ~8-12 s per image to accept/nudge a pre-drawn box -> ~2-3 hours of labeling for
  ~200 images including QA and the clipped cases. Training is trivial (minutes on CPU with
  yolo11n). Mirror the existing dataset_tools export format, add a `logo` class.

## Recommendation
1. Proceed — a logo class is feasible and cheaply labeled (~2-3 h). It cleanly handles
   >=5/7 residuals today and should reach ~6/7 once clipped wordmarks are hand-labeled
   (the a3dce37a generalization result predicts this).
2. Do NOT expect it to fix 572b0f91 — that leak is a bottom/tagline artifact, not a
   wordmark FP. Handle separately (or accept as a distinct known-limit).
3. Integrate as a GATED suppressor, not a blanket exclude: reject only letter-sized chunk
   components whose centroid lands in a predicted logo box, preserving the larger-mass
   exemption — same principle as today's dev_logo_band_suppress, but with a
   colour-agnostic, clipped-robust box source instead of the >=3-letter text-line rule.
4. Re-validate the full 92-set / 500-set after integration (ROI/threshold interactions).
