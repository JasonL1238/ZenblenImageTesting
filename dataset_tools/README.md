# Smoothie segmentation dataset pipeline

**Label once, export many.** One master annotation file is the source of truth;
scripts convert it into YOLO-seg, semantic-segmentation, and COCO datasets that
stay perfectly aligned (same images, same train/val/test split).

```
raw images
  → master polygon annotations   (master/annotations/smoothie_master.json)
  → generated binary mask PNGs    (master/masks/*.png)
  → YOLO-seg export               (exports/yolo_seg/)
  → semantic export               (exports/semantic/)
  → COCO export                   (exports/coco/)
```

## Class definition

One class only: **`0: smoothie`** (COCO category id `1`, YOLO class id `0`).

- Label only the **visible smoothie/liquid** pixels — never cup, lid, straw,
  hand, table, or background.
- If foam is treated as smoothie, include it **every time** (be consistent).
- Partially hidden smoothie → label only the **visible** region.
- Multiple smoothie regions in one image → label **all** of them.
- No smoothie visible → keep the image as an **intentional negative** (it stays
  in the master with zero annotations and exports as an empty label).

## Folder structure

```
smoothie_dataset/
  raw_images/                     # drop-zone for new source images
  master/
    images/                       # canonical copy of every labeled image
    annotations/smoothie_master.json   # ← SOURCE OF TRUTH (COCO polygons)
    masks/                        # generated 0/1 binary masks (one per image)
  splits/
    train.txt  val.txt  test.txt  # image stems, one per line (sticky)
  exports/
    yolo_seg/   images/{train,val,test}/  labels/{train,val,test}/  data.yaml
    semantic/   images/{train,val,test}/  masks/{train,val,test}/
    coco/       images/{train,val,test}/  annotations/instances_{train,val,test}.json
  debug_outputs/                  # visualize overlays land here
```

## Why label once, export many

Different model families want different label formats:

| Export     | Best for                                            | Label format            |
| ---------- | --------------------------------------------------- | ----------------------- |
| `yolo_seg` | **real-time** models (YOLO11-seg / YOLOv8-seg)      | polygon `.txt` (norm.)  |
| `semantic` | **smoothie-pixel** models (SegFormer, DeepLabV3, U-Net, BiSeNet, Fast-SCNN) | 0/1 mask PNG |
| `coco`     | **heavy cloud/research** models (Mask R-CNN, Mask2Former, Detectron2, MMDetection) | COCO polygon JSON |

Maintaining three hand-labeled datasets is a recipe for drift. Instead we keep
**one** master and regenerate the rest — fix a polygon once and every export
updates.

## Install

```bash
pip install -r requirements.txt        # opencv, numpy, pyyaml, pillow, tqdm, matplotlib
```

`pycocotools` is optional (only for RLE round-trips / COCO eval downstream).

## Quick start

```bash
# 0. create the folder skeleton
python dataset_pipeline.py init --dataset smoothie_dataset

# 1. build the master (pick ONE source):
#    a) from this repo's SQLite labeling tool (labeling/):
python dataset_pipeline.py import-labels --dataset smoothie_dataset \
       --db labeling/labels.db --images-dir labeling/data/images
#    b) or from an external COCO polygon file:
python dataset_pipeline.py import-coco --dataset smoothie_dataset \
       --coco path/to/annotations.json --images-dir path/to/images

# 2. validate before doing anything else
python dataset_pipeline.py validate --dataset smoothie_dataset

# 3. eyeball a sample of the labels
python dataset_pipeline.py visualize --dataset smoothie_dataset --num 20

# 4. fix the split once (sticky — reused by every export)
python dataset_pipeline.py create-splits --dataset smoothie_dataset \
       --train 0.8 --val 0.1 --test 0.1

# 5. export everything
python dataset_pipeline.py export-all --dataset smoothie_dataset
#    (or individually: export-yolo / export-semantic / export-coco)
```

## CLI reference

| Command           | What it does                                                     |
| ----------------- | ---------------------------------------------------------------- |
| `init`            | create the folder skeleton (`--scan-raw` seeds master from `raw_images/`) |
| `import-coco`     | build master from an external COCO polygon file                  |
| `import-labels`   | build master from the SQLite labeling tool (`labeling/`)         |
| `create-splits`   | assign + persist `train/val/test` (`--train/--val/--test`, `--seed`, `--force`) |
| `generate-masks`  | rasterize master polygons → `master/masks/*.png` (0/1)           |
| `export-yolo`     | write `exports/yolo_seg/` + `data.yaml`                          |
| `export-semantic` | write `exports/semantic/` (image + 0/1 mask pairs)               |
| `export-coco`     | write `exports/coco/` (`instances_{train,val,test}.json`)        |
| `export-all`      | `generate-masks` + all three exports                             |
| `validate`        | check annotations, polygon bounds, ≥3 points, empty masks, files |
| `visualize`       | write image+polygon+mask overlay panels to `debug_outputs/`      |

All commands take `--dataset <root>` (default `smoothie_dataset`).

## Splits are sticky

`create-splits` writes `splits/{train,val,test}.txt` once and every exporter reads
those files, so YOLO / semantic / COCO always share an identical partition.
Re-running `create-splits` **reuses** the existing split unless you pass `--force`
— so adding a new export never silently reshuffles your data.

## Adding new images

1. Drop images into `raw_images/` (or the labeling tool's image dir).
2. Annotate them (labeling UI → SQLite, or extend the COCO file).
3. Re-run the relevant `import-*` command to refresh the master.
4. `create-splits` (existing images keep their split; use `--force` only if you
   want a full reshuffle), then `export-all`.

## Inspecting labels

- `validate` — machine checks (bounds, point counts, empty masks, missing files).
- `visualize --num N` — writes N side-by-side panels (polygon outline | filled
  mask) to `debug_outputs/` so you can confirm the mask matches the smoothie.

## Regenerating exports

Exports are disposable — the master is the truth. Any time you change the master:

```bash
python dataset_pipeline.py export-all --dataset smoothie_dataset
```

Each export directory is rebuilt from scratch, so stale files never linger.

## Training after export

Prefer the multi-mode trainer for live weights:

```bash
/opt/miniconda3/bin/python training/train_multi.py --mode standard
```

A YOLO-seg starter also lives in [`training_examples/train_yolo_seg.py`](training_examples/train_yolo_seg.py)
(Ultralytics on `exports/yolo_seg/data.yaml`). For the older smoothie_dataset
path see `training/train.py`.
