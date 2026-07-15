# Zenblen Image Testing

Computer-vision pipeline for scoring smoothie **blendedness** — how uniformly ingredients are mixed inside the container.

## Quick start

```bash
# Single image
python run.py --pipeline classical --image data/images/test.jpg

# Batch (directory)
python run.py --pipeline classical --image data/images/ --threshold 0.90
```

## Pipeline

The **classical** CV pipeline scores blendedness by finding unblended chunks
inside the container. Container detection (ROI) uses a fine-tuned YOLO11n-seg
model as the priority detector with a classical colour-threshold fallback.
Chunk detection uses a trained YOLO chunk model with classical local-deviation
as fallback. See `CLAUDE.md` for the design.

Labeling + training live under `labeling/` and `training/train_multi.py`.

## Outputs

Each run writes to `outputs/<timestamp>__<pipeline>/`:

- `README.md` — summary and failures
- `comparison.csv` — per-image scores
- `run_info.json` — full run metadata
- `<shade>/` — mask and ROI overlays per image

Pass/fail is `blend_score >= threshold` (default 0.90).

## Tests

```bash
pytest smoothie_cv/tests/test_pipeline.py::TestClassicalCVPipeline -v
```
