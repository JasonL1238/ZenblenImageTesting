"""Train a YOLO-seg model for a multi-mode labeler dataset (spill / logo / standard / chunk).

Each mode is its own single-class YOLO11n-seg model, independent of the container
detector trained by training/train.py. Same recipe (nano net, CPU — MPS segfaults
on YOLO-seg), just pointed at the per-mode dataset produced by export_multi.py.

Run from the repo root (all paths below are cwd-relative). End-to-end, once images
are labeled in app_multi.py:
    python labeling/export_multi.py --mode spill              # build the dataset
    /opt/miniconda3/bin/python training/train_multi.py --mode spill   # train

Runs save to runs/<mode>-seg/<name>/weights/{best,last}.pt

Prereqs:
    - Conda base env has ultralytics + torch: /opt/miniconda3/bin/python
    - Base checkpoint yolo11n-seg.pt (ultralytics auto-downloads if missing)
    - Dataset exported for the mode (the --mode's data.yaml must exist)
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Per-mode config. `data` matches export_multi.py's output dirs; `deploy` is the
# suggested live-weights path (these are NEW models — they do NOT overwrite the
# container detector's checkpoints/yolo_smoothie_seg.pt).
MODE_CFG: dict[str, dict[str, str]] = {
    "standard": {
        "data":    "labeling/smoothie_dataset_std/data.yaml",
        "project": "runs/standard-seg",
        "deploy":  "checkpoints/yolo_standard_seg.pt",
    },
    "spill": {
        "data":    "labeling/spill_dataset/data.yaml",
        "project": "runs/spill-seg",
        "deploy":  "checkpoints/yolo_spill_seg.pt",
    },
    "logo": {
        "data":    "labeling/logo_dataset/data.yaml",
        "project": "runs/logo-seg",
        "deploy":  "checkpoints/yolo_logo_seg.pt",
    },
    "chunk": {
        "data":    "labeling/chunk_dataset/data.yaml",
        "project": "runs/chunk-seg",
        "deploy":  "checkpoints/yolo_chunk_seg.pt",
    },
}

BASE_MODEL = "yolo11n-seg.pt"   # nano; swap to yolo11s-seg.pt for more capacity
DEVICE = "cpu"                  # MPS (Apple Silicon) segfaults on YOLO-seg


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a per-mode YOLO-seg model")
    ap.add_argument("--mode", required=True, choices=list(MODE_CFG),
                    help="which labeler mode / dataset to train on")
    ap.add_argument("--name", help="run name (default: <mode>-nano-v1)")
    ap.add_argument("--base", default=BASE_MODEL, help=f"base weights (default: {BASE_MODEL})")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default=DEVICE, help="cpu | 0 | mps (mps unsupported for seg)")
    args = ap.parse_args()

    cfg = MODE_CFG[args.mode]
    run_name = args.name or f"{args.mode}-nano-v1"
    data_yaml = Path(cfg["data"])
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found — export the dataset first:\n"
            f"    python labeling/export_multi.py --mode {args.mode}"
        )

    # Import here so a missing dataset fails fast without loading torch.
    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project=cfg["project"],
        name=run_name,
        exist_ok=False,
    )

    metrics = model.val()
    best = f"{cfg['project']}/{run_name}/weights/best.pt"
    print(f"\n[{args.mode}] mAP50 (mask):    {metrics.seg.map50:.4f}")
    print(f"[{args.mode}] mAP50-95 (mask): {metrics.seg.map:.4f}")
    print(f"[{args.mode}] Weights: {best}")
    print(f"[{args.mode}] To deploy: cp {best} {cfg['deploy']}")


if __name__ == "__main__":
    main()
