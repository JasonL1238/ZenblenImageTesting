"""Command-line interface for the smoothie dataset pipeline.

Run via the repo-root entry point::

    python dataset_pipeline.py <command> [--dataset smoothie_dataset] [options]

Commands
    init             create the folder skeleton (+ optional master from raw_images)
    import-coco      build the master from an external COCO polygon file
    import-labels    build the master from the SQLite labeling tool (labeling/)
    create-splits    assign train/val/test and persist splits/*.txt
    generate-masks   rasterize master polygons -> master/masks/*.png
    export-yolo      write exports/yolo_seg/ (+ data.yaml)
    export-semantic  write exports/semantic/ (image + 0/1 mask pairs)
    export-coco      write exports/coco/ (instances_{train,val,test}.json)
    export-all       generate-masks + all three exports
    validate         check annotations, polygons, masks, files
    visualize        write overlay panels to debug_outputs/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import exporters, master as master_mod, splits as splits_mod, validate as validate_mod, visualize as visualize_mod
from .config import DatasetPaths


def _paths(args: argparse.Namespace) -> DatasetPaths:
    p = DatasetPaths.from_root(args.dataset)
    p.ensure()
    return p


def _load_master(paths: DatasetPaths) -> master_mod.Master:
    return master_mod.Master.load(paths.master_json)


# ---- command handlers -------------------------------------------------------
def cmd_init(args: argparse.Namespace) -> int:
    paths = _paths(args)
    print(f"dataset root: {paths.root}")
    if args.scan_raw:
        m = master_mod.scan_raw_images(paths, copy_images=True)
        m.save(paths.master_json)
        print(f"  master initialised from raw_images: {len(m.images)} images, 0 annotations")
    else:
        if not paths.master_json.exists():
            master_mod.Master().save(paths.master_json)
            print("  empty master written")
    print("  folder skeleton ready.")
    return 0


def cmd_import_coco(args: argparse.Namespace) -> int:
    paths = _paths(args)
    m = master_mod.import_from_coco(
        Path(args.coco), Path(args.images_dir), paths,
        allow_extra_categories=args.allow_extra_categories, copy_images=True,
    )
    m.save(paths.master_json)
    print(f"imported {len(m.images)} images, {len(m.annotations)} annotations "
          f"-> {paths.master_json}")
    return 0


def cmd_import_labels(args: argparse.Namespace) -> int:
    paths = _paths(args)
    verdicts = tuple(v.strip() for v in args.verdicts.split(",") if v.strip())
    m = master_mod.import_from_labeling_db(
        Path(args.db), Path(args.images_dir), paths,
        verdicts=verdicts, copy_images=True,
    )
    m.save(paths.master_json)
    n_neg = sum(1 for im in m.images if not m.annotations_for(im["id"]))
    print(f"imported {len(m.images)} images ({n_neg} negatives), "
          f"{len(m.annotations)} annotations -> {paths.master_json}")
    return 0


def cmd_create_splits(args: argparse.Namespace) -> int:
    paths = _paths(args)
    m = _load_master(paths)
    splits_mod.create_splits(
        m, paths, train=args.train, val=args.val, test=args.test,
        seed=args.seed, force=args.force,
    )
    return 0


def cmd_generate_masks(args: argparse.Namespace) -> int:
    paths = _paths(args)
    m = _load_master(paths)
    n = exporters.generate_masks(m, paths)
    print(f"wrote {n} masks -> {paths.master_masks}")
    return 0


def _require_splits(paths: DatasetPaths) -> bool:
    if not splits_mod.splits_exist(paths):
        print("error: no splits found. Run `create-splits` first.", file=sys.stderr)
        return False
    return True


def cmd_export_yolo(args: argparse.Namespace) -> int:
    paths = _paths(args)
    if not _require_splits(paths):
        return 1
    counts = exporters.export_yolo(_load_master(paths), paths)
    print(f"YOLO-seg export: {counts} -> {paths.yolo_dir}")
    return 0


def cmd_export_semantic(args: argparse.Namespace) -> int:
    paths = _paths(args)
    if not _require_splits(paths):
        return 1
    counts = exporters.export_semantic(_load_master(paths), paths)
    print(f"semantic export: {counts} -> {paths.semantic_dir}")
    return 0


def cmd_export_coco(args: argparse.Namespace) -> int:
    paths = _paths(args)
    if not _require_splits(paths):
        return 1
    counts = exporters.export_coco(_load_master(paths), paths)
    print(f"COCO export: {counts} -> {paths.coco_dir}")
    return 0


def cmd_export_all(args: argparse.Namespace) -> int:
    paths = _paths(args)
    if not _require_splits(paths):
        return 1
    exporters.export_all(_load_master(paths), paths)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    paths = _paths(args)
    rep = validate_mod.validate(
        _load_master(paths), paths,
        allow_extra_categories=args.allow_extra_categories,
        check_files=not args.no_check_files,
    )
    print(rep.summary())
    return 0 if rep.ok else 1


def cmd_visualize(args: argparse.Namespace) -> int:
    paths = _paths(args)
    n = visualize_mod.visualize(_load_master(paths), paths, num=args.num)
    print(f"wrote {n} overlays -> {paths.debug_dir}")
    return 0


# ---- parser -----------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dataset_pipeline.py",
        description="Smoothie segmentation dataset: label once, export many formats.",
    )
    parser.add_argument("--dataset", default="smoothie_dataset",
                        help="Dataset root directory (default: smoothie_dataset)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="create folder skeleton")
    p.add_argument("--scan-raw", action="store_true",
                   help="seed the master by scanning raw_images/ (annotations empty)")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("import-coco", help="build master from an external COCO file")
    p.add_argument("--coco", required=True, help="path to source COCO json")
    p.add_argument("--images-dir", required=True, help="dir holding the source images")
    p.add_argument("--allow-extra-categories", action="store_true",
                   help="fold all source categories into 'smoothie' instead of erroring")
    p.set_defaults(func=cmd_import_coco)

    p = sub.add_parser("import-labels", help="build master from the SQLite labeling tool")
    p.add_argument("--db", default="labeling/labels.db", help="path to labels.db")
    p.add_argument("--images-dir", default="labeling/data/images",
                   help="dir holding the labeled source images")
    p.add_argument("--verdicts", default="good,corrected",
                   help="comma list of verdicts to import (default: good,corrected)")
    p.set_defaults(func=cmd_import_labels)

    p = sub.add_parser("create-splits", help="assign + persist train/val/test")
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--val", type=float, default=0.1)
    p.add_argument("--test", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="reshuffle even if splits exist")
    p.set_defaults(func=cmd_create_splits)

    p = sub.add_parser("generate-masks", help="rasterize polygons -> master/masks")
    p.set_defaults(func=cmd_generate_masks)

    p = sub.add_parser("export-yolo", help="write exports/yolo_seg")
    p.set_defaults(func=cmd_export_yolo)
    p = sub.add_parser("export-semantic", help="write exports/semantic")
    p.set_defaults(func=cmd_export_semantic)
    p = sub.add_parser("export-coco", help="write exports/coco")
    p.set_defaults(func=cmd_export_coco)
    p = sub.add_parser("export-all", help="masks + yolo + semantic + coco")
    p.set_defaults(func=cmd_export_all)

    p = sub.add_parser("validate", help="check annotations/polygons/masks/files")
    p.add_argument("--allow-extra-categories", action="store_true")
    p.add_argument("--no-check-files", action="store_true",
                   help="skip on-disk image existence checks")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("visualize", help="write overlay panels to debug_outputs")
    p.add_argument("--num", type=int, default=20)
    p.set_defaults(func=cmd_visualize)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
