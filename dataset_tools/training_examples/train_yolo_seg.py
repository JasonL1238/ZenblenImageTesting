"""Example: fine-tune Ultralytics YOLO-seg on the exported dataset.

Prereq:
    pip install ultralytics
    python dataset_pipeline.py export-all --dataset smoothie_dataset

YOLO reads the polygon .txt labels and data.yaml produced by `export-yolo`.
Start from a small pretrained seg checkpoint (n = fastest, s = a bit stronger).
"""
from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

DATA_YAML = Path("smoothie_dataset/exports/yolo_seg/data.yaml")


def main() -> None:
    # yolo11n-seg.pt (nano, real-time) or yolo11s-seg.pt (small, a bit more accurate)
    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        patience=25,
        project="runs/smoothie-seg",
        name="yolo11n",
    )
    # Validate on the val split defined in data.yaml
    metrics = model.val()
    print("mAP50-95 (mask):", metrics.seg.map)

    # Predict on a new image:
    #   results = model.predict("some_cup.jpg", save=True)
    #   mask = results[0].masks.data  # (N, H, W) tensor of instance masks


if __name__ == "__main__":
    main()
