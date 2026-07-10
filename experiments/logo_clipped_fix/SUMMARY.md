# Clipped-wordmark logo FP fix — research + implementation

Goal: stop parts of the "zenblen" logo being flagged as unblended chunks, focused on
the CLIPPED/CURVED wordmark residual (partly out of frame / wraps around the cup → <3
letters or short span → defeats the ≥3-letter text-line detector AND EAST OCR).

## What the problem actually is
- 92-image curated set: already CLEAN (0 logo FPs).
- 500-image labeling-disjoint audit (`experiments/api_corpus_failure_audit/`): 24 logo
  FPs = the #1 failure mode, 22/24 red_pink. 8 hard-core cases defeat both the
  text-line detector and EAST: `27a1a9c2 2b205adb 6a12555a 877c3144 a3dce37a b4cf147c
  c197ebca dce9974f`.
- Leaked letters (318–663px) sit inside the real-chunk size range → color/size can't
  separate them.

## Step 1 — band prior: WIN (`band_prior/findings.md`)
The fixed rig prints the wordmark HIGH and it curves to the vertical EDGES, so clipped
fragments land in the top-LEFT/RIGHT CORNERS. Real chunks never sit above y_frac≈0.25
(median 0.52) and cluster centrally. Measured on 24 logo + 147 real-chunk components:
- rule `y_frac ≤ 0.30 AND edge_dist ≤ 0.25` → 16/24 logo FPs (8/11 hard components),
  **0/147 real chunks lost**.

## Step 2 — SWT/MSER letter primitive: RULED OUT (`swt_mser/`)
Stroke-width CV fully overlaps (logo 0.12–0.49 ⊂ real 0.03–0.64); topology/holes bust
(4/24 logo have a hole vs 7/147 chunks). The 4 compact dark/chroma fragments are
geometrically identical to small chunks. Best appearance rule reached 3/8 corner-misses
via `solidity` — a feature the pipeline already has. No new signal.

## Implemented — wordmark-gated top-corner gate (`dev_logo_corner_*`)
`smoothie_cv/config.py` + `smoothie_cv/pipelines/classical_cv.py` accept loop.
Veto an otherwise-accepted component whose centroid is in the top band (y_frac ≤ 0.30)
AND near a vertical edge (min(x_frac,1-x_frac) ≤ 0.25) — **only when NO wordmark was
confirmed** (`not logo_labels`).

The wordmark-gating is the crux: a first, ungated version regressed 2 documented real
chunks on the 92-set (8343d981, ac4eac46 — dark high-solidity chunks that sit in the
top-left corner, position-identical to logo fragments). BUT both have a CONFIRMED
wordmark, where the existing `logo_band` area-ceiling already spares them. The clipped
case this rule targets always has an UNconfirmed wordmark. Gating on `not logo_labels`
recovered both while keeping the clipped-case suppression.

## Verification
- 92-set `validate_chunks.py` (regression gate of record): **Δ0, 30 flagged, 0 new / 0
  lost** — zero regressions.
- End-to-end on the 8 hard cases (`verify_corner_gate.py`, real pipeline ON vs OFF):
  **6/8 suppressed** (2b205adb 2075→0, b4cf147c 1146→0, c197ebca 257→0, 6a12555a &
  877c3144 →0, 27a1a9c2 663→28 below flag). 2 unchanged (a3dce37a, dce9974f) —
  mid-frame fragments outside the corner zone.
- `pytest ...::TestClassicalCVPipeline` — 8/8 pass.

## Residual → needs Step A (trained YOLO logo-mask class)
Not reachable by any classical rule: (a) a3dce37a, dce9974f mid-frame fragments; (b) the
4 compact dark/chroma fragments (SWT/MSER-proof). Semantic detection is the only lever.
Open question for Jason: can the capture rig control cup orientation (Step F)? A
consistent orientation moots the whole clipped-wordmark class at the source.
