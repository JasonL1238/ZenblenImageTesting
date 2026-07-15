"""Train YOLO-seg container detector.

Run from the repo root (all paths below are cwd-relative):

Prereqs:
    - Conda base env: /opt/miniconda3/bin/python training/train.py
    - Dataset exported: python dataset_pipeline.py export-all
    - Base checkpoint present: yolo11n-seg.pt (download if missing)

Each run saves to runs/smoothie-seg/<name>/weights/{best,last}.pt

After training, evaluate then PROMOTE the weights into the live pipeline:
    python scripts/compare_yolo_vs_sam.py --weights runs/smoothie-seg/<name>/weights/best.pt
    cp runs/smoothie-seg/<name>/weights/best.pt checkpoints/yolo_smoothie_seg.pt
    rm -rf outputs/roi_cache_yolo && python scripts/cache_yolo_rois.py
    python scripts/validate_chunks.py      # re-validate all 92 before trusting it
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("smoothie_dataset/exports/yolo_seg/data.yaml")
BASE_MODEL = "yolo11n-seg.pt"   # nano; swap to yolo11s-seg.pt for more capacity
RUN_NAME   = "nano-v5"          # bump each retrain (v4 = previously deployed)

# MPS (Apple Silicon) segfaults with YOLO segmentation — always use CPU here.
DEVICE = "cpu"


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} not found — run: python dataset_pipeline.py export-all"
        )

    model = YOLO(BASE_MODEL)
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=30,
        device=DEVICE,
        project="runs/smoothie-seg",
        name=RUN_NAME,
        exist_ok=False,
    )

    metrics = model.val()
    print(f"\nmAP50 (mask):    {metrics.seg.map50:.4f}")
    print(f"mAP50-95 (mask): {metrics.seg.map:.4f}")
    print(f"Weights: runs/smoothie-seg/{RUN_NAME}/weights/best.pt")
    print(f"To deploy: cp runs/smoothie-seg/{RUN_NAME}/weights/best.pt "
          f"checkpoints/yolo_smoothie_seg.pt  (then re-cache ROIs + validate)")


if __name__ == "__main__":
    main()
