# Chunk-segmentation ablation

An **isolated, read-only** experiment to evaluate new methods for detecting
unblended **chunks inside an already-good smoothie mask**. It does **not**
re-segment the smoothie and does **not** touch the production pipeline.

- Reuses the production smoothie ROI (`outputs/roi_cache_yolo/<stem>.png`).
- Reads production verdicts (`outputs/report/scores.csv`) only for comparison.
- Writes everything under `experiments/chunk_segmentation_ablation/results/`.
- Never imports `smoothie_cv`; never writes outside this directory.

See `report.md` for results and the recommendation.

## Required inputs (already present in the repo)
| Input | Path | Notes |
|---|---|---|
| Source images | `data/images/{red_pink,yellow}/<stem>.jpg` | 92 images |
| Smoothie ROI masks | `outputs/roi_cache_yolo/<stem>.png` | binary 0/255, reused as-is |
| Production verdicts | `outputs/report/scores.csv` | baseline for comparison |
| Production overlays | `outputs/report/overlays/<stem>.png` | shown in gallery |

Python: `/opt/miniconda3/bin/python` (needs cv2, numpy, scipy, skimage — all present).

## How to run
```bash
PY=/opt/miniconda3/bin/python
cd <repo root>

# Run all methods on the full test set (bad + good), write per-method outputs
# and the aggregate results_summary.csv:
$PY experiments/chunk_segmentation_ablation/run_experiment.py

# Subsets:
$PY experiments/chunk_segmentation_ablation/run_experiment.py --set bad
$PY experiments/chunk_segmentation_ablation/run_experiment.py --methods residual_highpass dog_blob

# Build the self-contained HTML comparison gallery:
$PY experiments/chunk_segmentation_ablation/build_gallery.py
# -> open experiments/chunk_segmentation_ablation/gallery.html
```

## Test sets (`test_sets.py`)
- `BAD_CASES` / `BAD_CASE_NOTES` — images where production has trouble detecting
  chunks (misses, false positives, or unanalyzable), each annotated with why.
- `GOOD_CASES` — 50 previously-clean "do-not-break" cups (regression protection).

No pixel-level ground truth exists, so evaluation is comparison-based: production
verdict + visual overlay inspection + candidate-count sanity checks (see report.md).

## Methods (`methods/<name>.py`)
Each is independently runnable and satisfies a common contract
(`NAME`, `DEFAULT_PARAMS`, `detect(img_bgr, roi_mask, params) -> MethodResult`);
see the docstring in `common.py`. All use **adaptive per-image percentile
thresholds inside the ROI** — no fixed raw colour/intensity values.

| Module | Idea |
|---|---|
| `residual_highpass` | high-pass residual vs blurred liquid background |
| `dog_blob` | Difference-of-Gaussians / LoG multi-scale blobs |
| `texture_variance` | local variance / std texture map |
| `gradient_magnitude` | Sobel/Laplacian gradient-density map |
| `morphology_contour` | candidate + morphology/contour cleanup filtering |
| `watershed_split` | split merged/touching candidate blobs (splitter, not detector) |
| `ensemble` | normalized residual + variance + gradient, percentile-thresholded |

## Outputs
```
experiments/chunk_segmentation_ablation/
├── results/<method>/
│   ├── heatmap/<stem>.png         raw candidate heatmap (JET)
│   ├── mask/<stem>.png            binary chunk mask
│   ├── overlay/<stem>.png         contours on original
│   ├── overlay_in_roi/<stem>.png  contours inside dimmed ROI
│   └── json/<stem>.json           count, area stats, scores, rejects, params
├── results_summary.csv            one row per image per method
├── gallery.html                   side-by-side visual comparison
├── report.md                      results + recommendation
└── README.md                      this file
```
