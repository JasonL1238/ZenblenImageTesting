#!/usr/bin/env python3
"""
End-to-end eval harness for BOTH deployed pipelines (chunk + spill) on a
training-disjoint, smoothie-present sample of unlabeled frames.

Test set construction (all three conditions required):
  - on disk in labeling/data/images/<file_id>.jpg
  - NOT in any of the 3 models' train/val/test splits (smoothie_dataset_std,
    spill_dataset, logo_dataset) — no train/test leakage
  - smoothie-present: has a `standard` prediction row in labels.db (the machine
    dump is mostly empty-interior frames; this keeps the eval on real cups)
  - never hand-labeled in ANY mode (annotations/labels/mode_status) — truly unlabeled

For each frame it runs:
  - CHUNK: detect_container (standard YOLO ROI) → ClassicalCVPipeline (logo-YOLO
    suppression on) → blend_score / passed / mask
  - SPILL: SpillPipeline (yolo_spill_seg) → detected / area / mask

Outputs (outputs/pipeline_eval/<stamp>/):
  results.csv                 one row per frame, both pipelines
  summary.md                  aggregate stats
  composites/<id>.jpg         [original | chunk overlay | spill overlay] side-by-side
  review_chunks/<id>.jpg      symlinked composites for CHUNK-flagged frames
  review_spills/<id>.jpg      symlinked composites for SPILL-detected frames

Run:  /opt/miniconda3/bin/python scripts/eval_pipelines.py --n 300 [--seed 0]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smoothie_cv.config import Config
from smoothie_cv.detection import detect_container, draw_container_overlay
from smoothie_cv.pipelines.classical_cv import ClassicalCVPipeline
from smoothie_cv.pipelines.spill import SpillPipeline
from smoothie_cv.scoring.metrics import overlay_mask

REPO = Path(__file__).resolve().parent.parent
POOL = REPO / "labeling" / "data" / "images"
DB = REPO / "labeling" / "labels.db"
DATASETS = ["smoothie_dataset_std", "spill_dataset", "logo_dataset"]


def build_test_ids() -> list[int]:
    train_ids: set[int] = set()
    for d in DATASETS:
        for f in glob.glob(str(REPO / "labeling" / d / "images" / "*" / "*.jpg")):
            m = re.search(r"(\d+)\.jpg$", f)
            if m:
                train_ids.add(int(m.group(1)))
    on_disk = {int(p.stem) for p in POOL.glob("*.jpg") if p.stem.isdigit()}
    c = sqlite3.connect(DB)
    std_pred = {fid for (fid,) in c.execute(
        "select distinct file_id from predictions where mode='standard'")}
    labeled = set()
    for t in ("annotations", "labels", "mode_status"):
        labeled |= {fid for (fid,) in c.execute(f"select distinct file_id from {t}")}
    c.close()
    return sorted((std_pred & on_disk) - train_ids - labeled)


def side_by_side(*imgs: np.ndarray) -> np.ndarray:
    h = max(i.shape[0] for i in imgs)
    resized = []
    for i in imgs:
        if i.shape[0] != h:
            scale = h / i.shape[0]
            i = cv2.resize(i, (int(i.shape[1] * scale), h))
        resized.append(i)
    return np.hstack(resized)


def label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of frames to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default outputs/pipeline_eval/<stamp>)")
    args = ap.parse_args()

    ids = build_test_ids()
    print(f"test-eligible frames (disjoint + smoothie-present + unlabeled): {len(ids)}")
    random.seed(args.seed)
    sample = sorted(random.sample(ids, min(args.n, len(ids))))
    print(f"sampling {len(sample)} (seed={args.seed})")

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(args.out) if args.out else REPO / "outputs" / "pipeline_eval" / stamp
    (out / "composites").mkdir(parents=True, exist_ok=True)
    (out / "review_chunks").mkdir(parents=True, exist_ok=True)
    (out / "review_spills").mkdir(parents=True, exist_ok=True)

    cfg = Config()
    chunk = ClassicalCVPipeline(cfg)
    spill = SpillPipeline(cfg)

    rows = []
    t_start = time.perf_counter()
    for i, fid in enumerate(sample):
        p = POOL / f"{fid}.jpg"
        img = cv2.imread(str(p))
        if img is None:
            continue
        # chunk
        roi_mask, _b, det = detect_container(img, cfg, return_meta=True)
        cres = chunk.analyze(img, roi_mask)
        # spill
        sres = spill.analyze(img)

        # "chunks detected" = the chunk mask is non-empty (any unblended pixel) —
        # this is the project's real signal (matches scripts/validate_chunks.py).
        # cres.passed uses the 0.90 blend_score gate, which requires >10% of the
        # ROI to be chunk-area and therefore almost never trips; it's reported
        # separately, not used as the "found a chunk" flag.
        chunk_flagged = bool(cres.mask.any())

        roi_ov = draw_container_overlay(img, roi_mask)
        chunk_ov = overlay_mask(roi_ov, cres.mask)
        spill_ov = overlay_mask(img, sres.mask)
        comp = side_by_side(
            label(img, f"{fid} orig"),
            label(chunk_ov, f"chunk {cres.blend_score:.3f} {'CHUNKS' if chunk_flagged else 'clean'} det={det['detector']}",
                  (0, 0, 255) if chunk_flagged else (0, 255, 0)),
            label(spill_ov, f"spill {'DETECTED' if sres.spill_detected else 'clean'} {sres.spill_area_px}px c={sres.confidence:.2f}",
                  (0, 0, 255) if sres.spill_detected else (0, 255, 0)),
        )
        comp_path = out / "composites" / f"{fid}.jpg"
        cv2.imwrite(str(comp_path), comp)
        if chunk_flagged:
            try:
                (out / "review_chunks" / f"{fid}.jpg").symlink_to(comp_path)
            except FileExistsError:
                pass
        if sres.spill_detected:
            try:
                (out / "review_spills" / f"{fid}.jpg").symlink_to(comp_path)
            except FileExistsError:
                pass

        rows.append({
            "file_id": fid,
            "chunk_score": round(cres.blend_score, 4),
            "chunk_flagged": chunk_flagged,      # any chunk pixel found (real signal)
            "chunk_passed_090": cres.passed,     # score >= 0.90 gate (rarely trips)
            "chunk_detector": det["detector"],
            "chunk_fallback": det["fallback"],
            "spill_detected": sres.spill_detected,
            "spill_area_px": sres.spill_area_px,
            "spill_conf": round(sres.confidence, 3),
            "spill_n_inst": sres.metadata["n_instances"],
        })
        if (i + 1) % 25 == 0:
            dt = time.perf_counter() - t_start
            print(f"  {i+1}/{len(sample)}  ({dt/(i+1)*1000:.0f} ms/img)")

    # write csv
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n = len(rows)
    chunk_flag = sum(1 for r in rows if r["chunk_flagged"])
    chunk_fail = sum(1 for r in rows if not r["chunk_passed_090"])
    spill_det = sum(1 for r in rows if r["spill_detected"])
    fallback = sum(1 for r in rows if r["chunk_fallback"])
    scores = [r["chunk_score"] for r in rows]
    summary = [
        f"# Pipeline eval — {stamp}",
        "",
        f"- frames: **{n}** (disjoint + smoothie-present + unlabeled, seed={args.seed})",
        f"- container: `{cfg.yolo_weights.name}`  ·  logo-suppress: {cfg.dev_logo_yolo_suppress}",
        f"- spill: `{cfg.spill_weights.name}`  ·  min_area={cfg.spill_min_area_px}px  conf={cfg.spill_conf}",
        "",
        "## Chunk pipeline",
        f"- **chunks detected** (mask non-empty, real signal): **{chunk_flag}/{n}** ({chunk_flag/n*100:.1f}%)",
        f"- FAIL @ 0.90 blend gate (rarely trips): {chunk_fail}/{n}",
        f"- container fallback (non-YOLO): {fallback}/{n}",
        f"- score range {min(scores):.3f}–{max(scores):.3f}, mean {sum(scores)/n:.3f}",
        "",
        "## Spill pipeline",
        f"- spill DETECTED: **{spill_det}/{n}** ({spill_det/n*100:.1f}%)",
        "",
        "→ review_chunks/ (chunks found) and review_spills/ hold composites for review.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\nOutput → {out}")


if __name__ == "__main__":
    main()
