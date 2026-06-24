#!/usr/bin/env python3
"""
Smoothie blendedness pipeline runner.

Usage (single image):
    python run.py --pipeline classical --image data/images/test.jpg

Usage (all pipelines, single image):
    python run.py --pipeline all --image data/images/test.jpg

Usage (batch — directory of images):
    python run.py --pipeline classical --image data/images/

Outputs per run:
    outputs/<stem>_<pipeline>_mask.png      - unblended region overlay
    outputs/<stem>_<pipeline>_roi.png       - detected container boundary
    outputs/<stem>_<pipeline>_result.json   - score, passed, metadata

Batch comparison (--pipeline all on a directory):
    outputs/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent))

from smoothie_cv.config import Config
from smoothie_cv.detection import (
    detect_container,
    draw_container_overlay,
    _classify_smoothie,
    SmoothieType,
    YellowRefineParams,
)
from smoothie_cv.scoring.metrics import overlay_mask


PIPELINE_NAMES = ["classical", "vlm", "sam"]


def load_pipeline(name: str, config: Config):
    if name == "classical":
        from smoothie_cv.pipelines.classical_cv import ClassicalCVPipeline
        return ClassicalCVPipeline(config)
    if name == "vlm":
        from smoothie_cv.pipelines.vlm import VLMPipeline
        return VLMPipeline(config)
    if name == "sam":
        from smoothie_cv.pipelines.sam import SAMPipeline
        return SAMPipeline(config)
    raise ValueError(f"Unknown pipeline: {name!r}. Choose from {PIPELINE_NAMES}")


def _shade_subfolder(smoothie_type: SmoothieType) -> str:
    """Map smoothie type to output subfolder name."""
    if smoothie_type == SmoothieType.RED_PINK:
        return "red_pink"
    return "yellow"


def run_single(
    image_path: Path,
    pipeline_name: str,
    config: Config,
    detector: str | None = None,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    yellow_params = YellowRefineParams(
        erode_scale=config.yellow_erode_scale,
        delta_b=config.yellow_delta_b,
        a_max=config.yellow_a_max,
        L_max=config.yellow_L_max,
        chroma_min=config.yellow_chroma_min,
    )
    smoothie_type = _classify_smoothie(image)

    # ROI detection via the SAM-priority / classical-fallback dispatcher.
    # `detector` (None = auto priority) maps to the dispatcher's `prefer`.
    roi_mask, _bbox, det = detect_container(
        image, config, prefer=detector, yellow_params=yellow_params, return_meta=True
    )

    pipeline = load_pipeline(pipeline_name, config)

    t0 = time.perf_counter()
    result = pipeline.analyze(image, roi_mask)
    runtime_ms = (time.perf_counter() - t0) * 1000

    # One subfolder per smoothie, grouped by shade: <run>/<shade>/<stem>/
    shade_dir = config.output_dir / _shade_subfolder(smoothie_type)
    smoothie_dir = shade_dir / image_path.stem
    smoothie_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{image_path.stem}_{pipeline_name}"

    # ROI overlay from the chosen detector
    roi_path = smoothie_dir / f"{stem}_roi.png"
    cv2.imwrite(str(roi_path), draw_container_overlay(image, roi_mask))

    # mask overlay (unblended region highlighted), computed on the ROI
    mask_vis = overlay_mask(image, result.mask)
    mask_path = smoothie_dir / f"{stem}_mask.png"
    cv2.imwrite(str(mask_path), mask_vis)

    # write JSON result
    record = {
        "image": str(image_path),
        "pipeline": pipeline_name,
        "smoothie_type": smoothie_type.value,
        "blend_score": round(result.blend_score, 4),
        "passed": result.passed,
        "threshold": config.threshold,
        "runtime_ms": round(runtime_ms, 1),
        "detector": det["detector"],
        "detector_fallback": det["fallback"],
        "top_roughness": det["roughness"],
        "roi_path": str(roi_path),
        "mask_path": str(mask_path),
        "metadata": result.metadata,
    }
    json_path = smoothie_dir / f"{stem}_result.json"
    json_path.write_text(json.dumps(record, indent=2))

    status = "PASS" if result.passed else "FAIL"
    det_tag = f"  det={det['detector']}" + ("(fallback)" if det["fallback"] else "")
    print(
        f"[{pipeline_name}] {image_path.name}  "
        f"score={result.blend_score:.3f}  {status}  "
        f"({runtime_ms:.0f} ms){det_tag}"
    )
    print(f"  dir   → {smoothie_dir}")
    return record


def run_batch(
    image_dir: Path,
    pipeline_names: list[str],
    config: Config,
    detector: str | None = None,
) -> list[dict]:
    images = sorted(image_dir.rglob("*.jpg")) + sorted(image_dir.rglob("*.png"))
    if not images:
        print(f"No .jpg/.png images found in {image_dir}")
        return []

    rows: list[dict] = []
    for img_path in images:
        for pname in pipeline_names:
            try:
                row = run_single(img_path, pname, config, detector=detector)
                rows.append(row)
            except NotImplementedError as e:
                print(f"[{pname}] SKIP — {e}")
            except Exception as e:
                print(f"[{pname}] ERROR on {img_path.name}: {e}")
    return rows


def make_run_dir(output_root: Path, pipeline_names: list[str], started: datetime) -> Path:
    """Create a unique timestamped subfolder for this run and return it."""
    stamp = started.strftime("%Y-%m-%d_%H-%M-%S")
    label = "all" if len(pipeline_names) > 1 else pipeline_names[0]
    run_dir = output_root / f"{stamp}__{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def format_run_readme(info: dict) -> str:
    """Build a short human-readable summary; full data lives in comparison.csv."""
    run_id = info["run_id"]
    pipelines = ", ".join(info["pipelines"])
    multi_pipeline = len(info["pipelines"]) > 1
    summary = info["summary"]
    started = datetime.fromisoformat(info["started"]).strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# {run_id}",
        "",
        f"{pipelines} · `{info['image_source']}` · threshold **{info['threshold']:.2f}** · "
        f"{info['duration_s']}s · {started}",
        "",
        f"**{summary['pass']} pass / {summary['fail']} fail** "
        f"({info['num_images']} images) · "
        f"scores {summary['min_score']:.3f}–{summary['max_score']:.3f}, "
        f"mean {summary['mean_score']:.3f}",
    ]

    failures = [r for r in info["results"] if not r.get("passed")]
    if failures:
        lines.extend(["", "## Failures", ""])
        if multi_pipeline:
            lines.extend(["| Image | Pipeline | Score |", "|---|---|---|"])
            for r in failures:
                lines.append(
                    f"| `{Path(r['image']).name}` | {r['pipeline']} | {r['blend_score']:.3f} |"
                )
        else:
            lines.extend(["| Image | Score |", "|---|---|"])
            for r in failures:
                lines.append(f"| `{Path(r['image']).name}` | {r['blend_score']:.3f} |")

    lines.extend(["", "→ [comparison.csv](comparison.csv) · [run_info.json](run_info.json)", ""])
    return "\n".join(lines)


def write_run_manifest(
    run_dir: Path,
    records: list[dict],
    pipeline_names: list[str],
    image_source: Path,
    config: Config,
    started: datetime,
) -> None:
    """Write run_info.json (machine) + README.md (human) describing this run."""
    finished = datetime.now()
    passed = sum(1 for r in records if r.get("passed"))
    failed = len(records) - passed
    scores = [r["blend_score"] for r in records] or [0.0]

    info = {
        "run_id": run_dir.name,
        "started": started.isoformat(timespec="seconds"),
        "finished": finished.isoformat(timespec="seconds"),
        "duration_s": round((finished - started).total_seconds(), 2),
        "pipelines": pipeline_names,
        "threshold": config.threshold,
        "image_source": str(image_source),
        "num_images": len({r["image"] for r in records}),
        "num_results": len(records),
        "summary": {
            "pass": passed,
            "fail": failed,
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "mean_score": round(sum(scores) / len(scores), 4),
        },
        "results": records,
    }
    (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
    (run_dir / "README.md").write_text(format_run_readme(info))

    if records:
        csv_path = run_dir / "comparison.csv"
        fieldnames = ["image", "pipeline", "blend_score", "passed", "threshold", "runtime_ms"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoothie blendedness pipeline runner")
    parser.add_argument("--pipeline", required=True,
                        help=f"Pipeline name or 'all'. Choices: {PIPELINE_NAMES}")
    parser.add_argument("--image", required=True,
                        help="Path to an image file or directory of images")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Pass/fail threshold (0–1). Overrides config default (0.90).")
    parser.add_argument("--config", default=None,
                        help="Path to config.yaml (optional)")
    parser.add_argument("--output-dir", default=None,
                        help="Directory to write outputs (default: outputs/)")
    parser.add_argument("--detector", choices=["auto", "sam", "classical"], default="auto",
                        help="ROI detector. 'auto' = SAM priority, classical fallback "
                             "(default). 'sam'/'classical' force one.")
    args = parser.parse_args()

    config = Config.load(args.config)
    if args.threshold is not None:
        config.threshold = args.threshold
    detector = None if args.detector == "auto" else args.detector
    output_root = Path(args.output_dir) if args.output_dir is not None else config.output_dir

    image_path = Path(args.image)
    pipeline_names = PIPELINE_NAMES if args.pipeline == "all" else [args.pipeline]

    if not (image_path.is_dir() or image_path.is_file()):
        print(f"Error: {image_path} is not a file or directory.")
        sys.exit(1)

    # every invocation gets its own timestamped subfolder under outputs/
    started = datetime.now()
    run_dir = make_run_dir(output_root, pipeline_names, started)
    config.output_dir = run_dir
    print(f"Run directory → {run_dir}\n")

    records: list[dict] = []
    if image_path.is_dir():
        records = run_batch(image_path, pipeline_names, config, detector=detector)
    else:
        for pname in pipeline_names:
            try:
                records.append(run_single(image_path, pname, config, detector=detector))
            except NotImplementedError as e:
                print(f"[{pname}] SKIP — {e}")

    write_run_manifest(run_dir, records, pipeline_names, image_path, config, started)
    print(f"\nManifest → {run_dir / 'run_info.json'}")
    print(f"Summary  → {run_dir / 'README.md'}")


if __name__ == "__main__":
    main()
