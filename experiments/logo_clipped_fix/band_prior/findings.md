# Step 1 — Band-prior separability: VERDICT = **YES, strongly** (unexpected)

Measured on the 500-corpus audit: 24 confirmed logo-FP components vs 147 real-chunk
components (from `suspected_real_chunks/`). y_frac/x_frac are normalized to each ROI's
own top/bottom (0 = ROI top, 1 = ROI bottom); edge_dist = min(x_frac, 1-x_frac).

Data: `band_components.csv` (per-component), `per_image.csv`, `band_scatter.png`.
Reproduce: `/opt/miniconda3/bin/python experiments/logo_clipped_fix/band_prior/analyze.py`

## The rig prints the logo HIGH and toward the EDGES — real chunks don't go there

| class | y_frac med (q25–q75) | edge_dist med | note |
|---|---|---|---|
| logo FP (all 24) | **0.271** (0.24–0.30) | 0.161 | top third, edge-skewed |
| logo FP (8 hard-core) | **0.252** (0.24–0.29) | 0.124 | top **corners** |
| real chunk (147) | 0.522 (0.39–0.76) | 0.320 | central + lower, **min y=0.249** |

Real-chunk y_frac never drops below 0.249. Logo FPs pile up at 0.18–0.30. The
scatter (`band_scatter.png`) shows logo FPs occupy the top-left + top-right CORNERS,
an L-shaped zone real chunks avoid.

## Best rule found (hard exclusion, zero collateral)

`reject if y_frac <= 0.30 AND edge_dist <= 0.25`

| rule | logo caught | hard caught | REAL chunks lost |
|---|---|---|---|
| y≤0.30 & edge≤0.25 | **16/24** | **8/11** | **0/147** |
| y≤0.30 & edge≤0.22 | 14/24 | 8/11 | 0/147 |
| y≤0.32 & edge≤0.22 | 15/24 | 8/11 | 1/147 (a8321fc1, uses compact+dark) |
| y≤0.32 & edge≤0.22 & compact-path-only | 12/24 | 7/11 | 0/147 |

The single borderline real chunk (`a8321fc1`, y=0.311, edge≈0.14, area 533) fires via
`compact+dark` — the **compact-path-only** condition protects it while still catching
7/11 hard cases at zero cost. Logo FPs leak overwhelmingly via the **compact** path
(14/24; 8/11 hard), whereas real chunks spread across all paths and are often
multi-path — so "compact-only + top-corner" is a tight signature.

## Recommendation

**Implement as a new component gate** in `_deviation_mask` (config-flagged, e.g.
`dev_logo_corner_suppress`): reject an accepted component whose centroid is in the
top band AND near a vertical edge (optionally compact-path-only). Non-destructive: it
suppresses one component, does not touch the ROI or other detections. This is
independent of, and complementary to, the confirmed-wordmark band suppression (which
needs ≥3 aligned letters this rule does not).

Catches ~67% of all logo FPs and 8/11 hard-core clipped/curved components at zero
measured real-chunk loss — far better than the "marginal soft prior" originally
predicted.

## Limits / next
- 8/24 logo FPs sit OUTSIDE the corners (e.g. a low mid-cup fragment at y=0.71,
  and near-center top fragments) — NOT caught by geometry. Those need the SWT/MSER
  letter primitive (Step 2) or the trained YOLO logo class (Step A).
- MUST validate on the 92-set (`scripts/validate_chunks.py`) before committing — a
  y/edge gate can shift borderline verdicts. Measured 0 real loss here, but the 92-set
  is the regression gate of record.
- Thresholds (0.30 / 0.25) are from n=171 components; widen only with 92-set + full-500
  re-validation.
