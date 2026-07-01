"""Example: train Mask R-CNN on the `coco` export with Detectron2.

The `export-coco` command writes standard COCO instance-segmentation files:

    smoothie_dataset/exports/coco/
      images/{train,val,test}/<file_name>
      annotations/instances_{train,val,test}.json   # category 1 = smoothie

That is exactly what Detectron2, MMDetection, and Mask2Former expect, so the same
files work across all three. Below is a minimal Detectron2 setup.

Prereq:
    pip install torch torchvision
    pip install 'git+https://github.com/facebookresearch/detectron2.git'
    python dataset_pipeline.py export-coco --dataset smoothie_dataset
"""
from __future__ import annotations

from pathlib import Path

COCO = Path("smoothie_dataset/exports/coco")


def main() -> None:
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2 import model_zoo

    for split in ("train", "val"):
        register_coco_instances(
            f"smoothie_{split}", {},
            str(COCO / "annotations" / f"instances_{split}.json"),
            str(COCO / "images" / split),
        )

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("smoothie_train",)
    cfg.DATASETS.TEST = ("smoothie_val",)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # smoothie only
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 3000
    cfg.OUTPUT_DIR = "runs/smoothie-maskrcnn"

    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # MMDetection / Mask2Former: point their COCO dataset config at the same
    # instances_*.json + images/ folders; set num_classes = 1.


if __name__ == "__main__":
    main()
