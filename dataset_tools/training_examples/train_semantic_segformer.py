"""Example: train a semantic segmentation model on the `semantic` export.

The `export-semantic` command writes image + single-channel mask pairs:

    smoothie_dataset/exports/semantic/
      images/{train,val,test}/<stem>.jpg
      masks/{train,val,test}/<stem>.png     # pixel 0 = background, 1 = smoothie

That layout is model-agnostic — SegFormer, DeepLabV3, U-Net, BiSeNet, and
Fast-SCNN all consume image+mask pairs. Below is a minimal Hugging Face SegFormer
loop showing how to wire the pairs into a Dataset. Swap the model/framework
freely; only the data-loading contract matters.

Prereq:
    pip install torch torchvision transformers datasets pillow numpy
    python dataset_pipeline.py export-semantic --dataset smoothie_dataset
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

SEM = Path("smoothie_dataset/exports/semantic")
NUM_LABELS = 2  # 0 = background, 1 = smoothie


class SmoothieSemanticDataset:
    """Pairs each image with its 0/1 mask. Framework-agnostic accessor."""

    def __init__(self, split: str) -> None:
        self.img_dir = SEM / "images" / split
        self.msk_dir = SEM / "masks" / split
        self.stems = sorted(p.stem for p in self.img_dir.glob("*"))

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, i: int) -> tuple[Image.Image, np.ndarray]:
        stem = self.stems[i]
        image = Image.open(self.img_dir / f"{stem}.jpg").convert("RGB")
        mask = np.array(Image.open(self.msk_dir / f"{stem}.png"))  # values in {0, 1}
        return image, mask


def main() -> None:
    # --- Hugging Face SegFormer sketch -------------------------------------
    # from transformers import (SegformerForSemanticSegmentation,
    #                           SegformerImageProcessor, TrainingArguments, Trainer)
    #
    # processor = SegformerImageProcessor(do_reduce_labels=False)
    # model = SegformerForSemanticSegmentation.from_pretrained(
    #     "nvidia/segformer-b0-finetuned-ade-512-512",
    #     num_labels=NUM_LABELS, ignore_mismatched_sizes=True,
    # )
    # ... build torch Datasets from SmoothieSemanticDataset, then Trainer.train()
    #
    # For DeepLabV3 (torchvision), U-Net, BiSeNet, Fast-SCNN the same pairs feed a
    # standard pixel-wise cross-entropy loop; the only change is the model class.
    train = SmoothieSemanticDataset("train")
    val = SmoothieSemanticDataset("val")
    print(f"semantic dataset ready: train={len(train)} val={len(val)}, "
          f"{NUM_LABELS} classes (0=bg, 1=smoothie)")


if __name__ == "__main__":
    main()
