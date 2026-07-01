# SAM segmentation labeling tool

A mostly-standalone pipeline to build a **supervised container-segmentation
dataset**. Pull production images from the Zenblen Files API, run SAM to get
candidate masks, review/correct them in a local web UI, and export a labeled
dataset. Reuses the existing detector (`smoothie_cv.detection.detect_container`)
but is otherwise independent of the analysis pipeline.

## Stages

```
1. download.py  Files API  -> labeling/data/images/<file_id>.jpg   (+ files table)
2. run_sam.py   SAM        -> data/masks_sam/*.png + data/polygons_sam/*.json
3. app.py       Flask UI   -> accept / reject / correct  (verdicts in labels.db)
4. export.py               -> labeling/dataset/ (images, masks, labels.csv, manifest.json)
```

Stages are separate so the expensive SAM pass runs **once, offline**, and the UI
just serves precomputed results — fast enough to label thousands.

## Setup

```bash
# Run ALL commands from the REPO ROOT (ZenblenImageTesting/), not from labeling/.
# SAM resolves checkpoints/ relative to cwd, so running from the wrong directory
# will produce a FileNotFoundError.

pip install -r requirements.txt             # adds flask + requests
# API key goes in .env at repo root (already git-ignored):
echo "ZENBLEN_API_KEY=..." >> .env          # never commit the key
```

SAM (stage 2 only) needs the conda base env with torch + sam2 and a checkpoint in
`checkpoints/` (see CLAUDE.md). Stages 1/3/4 do **not** need torch.

## Usage

```bash
# All commands run from repo root:

# 1. Download every image/jpg in a time range (start with a NARROW range to test).
python labeling/download.py --start '2026-06-29 00:00:00' --end '2026-06-30 00:00:00'
#    optional: --category CleanDone   --type image/jpg   --list-only

# 2. Run SAM (in the conda env). --limit N for a quick test batch.
/opt/miniconda3/bin/python labeling/run_sam.py --limit 20

# 3. Label. Open http://127.0.0.1:5000
python labeling/app.py

# 4. Export the dataset (both classes).
python labeling/export.py --split 0.15
```

## Labeling shortcuts

| key | action |
|-----|--------|
| `A` | accept — mask is good (`good`, or `corrected` if you edited it) |
| `R` | reject — segmentation failed (`bad`) |
| `S` | skip / unsure |
| `E` | toggle edit mode |
| drag vertex | move a polygon point |
| click edge | insert a vertex |
| right-click vertex / `Del` | remove a vertex |
| `Z` | undo last edit |
| `←` / `→` | previous / next without saving |

Editing the polygon auto-marks the verdict `corrected` on accept. Verdicts persist
to `labels.db`, so closing and relaunching resumes at the first unlabeled image.

## Dataset output

- `dataset/images/<id>.jpg` — every non-skip labeled image.
- `dataset/masks/<id>.png` — rasterized final polygon for `good`/`corrected`;
  SAM's original (wrong) mask for `bad`.
- `dataset/labels.csv` — `file_id, order_id, verdict, corrected, created_at`.
- `dataset/manifest.json` — counts + provenance.
- `dataset/{train,val}.txt` — positive-set id lists (only with `--split`).

Positives (`good` + `corrected`) are clean segmentation ground truth; `bad` rows
let you also train a "did SAM succeed" quality classifier.
