"""dataset_tools — label once, export many.

A small framework that treats a single COCO-style *master* annotation file as the
source of truth for a one-class ("smoothie") instance-segmentation dataset, then
converts it into every format a downstream model needs:

    raw images
      -> master polygon annotations   (master/annotations/smoothie_master.json)
      -> generated binary mask PNGs    (master/masks/*.png)
      -> YOLO-seg export               (exports/yolo_seg/)
      -> semantic-segmentation export  (exports/semantic/)
      -> COCO export                   (exports/coco/)

The command-line entry point lives in ``dataset_pipeline.py`` at the repo root;
it simply calls :func:`dataset_tools.cli.main`.
"""
from __future__ import annotations

__all__ = ["config", "geometry", "master", "splits", "exporters", "validate", "visualize"]
