# Zenblen Image Testing

Two top-level folders only:

| Folder | Purpose |
|--------|---------|
| `active_pipeline/` | Deployable YOLO runtime (blend/chunk + spill) + weights |
| `training/` | Labeling UI, datasets, `train_multi.py`, training runs |

## Runtime (Jetson / inference)

```bash
cd active_pipeline
python run.py --pipeline blend --image <img.jpg>
python run.py --pipeline spill --image <img.jpg>
```

## Label + train

```bash
cd training
python labeling/app_multi.py
python labeling/export_multi.py --mode chunk
/opt/miniconda3/bin/python train_multi.py --mode chunk
# deploy:
cp runs/chunk-seg/<run>/weights/best.pt ../active_pipeline/checkpoints/yolo_chunk_seg.pt
cp runs/chunk-seg/<run>/weights/best.pt checkpoints/yolo_chunk_seg.pt   # labeling mirror
```

Logo weights stay under `training/checkpoints/` (not shipped in `active_pipeline`).
