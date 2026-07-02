"""Train YOLO-seg container detector.

Prereqs:
    - Conda base env: /opt/miniconda3/bin/python train.py
    - Dataset exported: python dataset_pipeline.py export-all
    - Base checkpoint present: yolo11n-seg.pt (download if missing)

Each run saves to runs/smoothie-seg/<name>/weights/{best,last}.pt
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("smoothie_dataset/exports/yolo_seg/data.yaml")
BASE_MODEL = "yolo11n-seg.pt"   # nano; swap to yolo11s-seg.pt for more capacity
RUN_NAME   = "nano-v3"          # bump each retrain

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


if __name__ == "__main__":
    main()
