# Zenblen Image Testing

Computer-vision pipeline for scoring smoothie **blendedness** — how uniformly ingredients are mixed inside the container.

## Quick start

```bash
# Single image
python run.py --pipeline classical --image data/images/test.jpg

# Batch (directory)
python run.py --pipeline classical --image data/images/

# Compare all pipelines
python run.py --pipeline all --image data/images/ --threshold 0.90
```

## Pipelines

| Pipeline | Notes |
|---|---|
| `classical` | OpenCV variance + Canny edges (default, no extra deps) |
| `vlm` | Claude vision API — requires `ANTHROPIC_API_KEY` |
| `sam` | SAM2 segmentation — requires checkpoint in `checkpoints/` |
| `segformer` | HuggingFace SegFormer model |

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
