# Project: ZenblenImageTesting

# Layout
- `active_pipeline/` — ONLY runtime copy of `smoothie_cv` + deploy weights + `run.py`
- `training/` — labeling (`training/labeling/`) + `train_multi.py` + `runs/` + training checkpoints

Do not recreate a root-level `smoothie_cv/` — that duplicates the active pipeline.

# Commands
## Runtime (from `active_pipeline/`)
- `python run.py --pipeline blend --image <img.jpg>`
- `python run.py --pipeline spill --image <img.jpg>`
- Test: `pytest smoothie_cv/tests/test_pipeline.py::TestBlendPipeline -v`

## Train (from `training/`)
- Label: `python labeling/app_multi.py`
- Export: `python labeling/export_multi.py --mode standard|spill|logo|chunk`
- Train: `/opt/miniconda3/bin/python train_multi.py --mode <mode>`
- Deploy standard/spill/chunk:
  `cp runs/<mode>-seg/<run>/weights/best.pt ../active_pipeline/checkpoints/yolo_<mode>_seg.pt`
  (also mirror into `training/checkpoints/` for labeling predict/flag)
- Logo deploy: `cp runs/logo-seg/<run>/weights/best.pt checkpoints/yolo_logo_seg.pt`

# Active path
YOLO-only: standard-seg ROI → chunk-seg → blend score; spill-seg full-frame.

# Session health (canary)
- Begin EVERY response with the marker 🟢 followed by a space.

# Compaction
- Preserve: modified files, approach + WHY, test/run commands.
