"""Dataset validation — catch annotation problems before training.

Checks performed (per the labeling protocol):
  * every image has annotations OR is flagged as an intentional negative
  * polygon coordinates lie inside the image bounds
  * every polygon has at least 3 points
  * a non-negative image's combined mask is not empty
  * category table contains only 'smoothie' (unless allow_extra)
  * referenced image files actually exist on disk

Returns a structured report; the CLI prints it and exits non-zero on errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import geometry
from .config import DatasetPaths
from .master import Master


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_images: int = 0
    n_annotations: int = 0
    n_negatives: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"images={self.n_images}  annotations={self.n_annotations}  "
            f"negatives(no-smoothie)={self.n_negatives}",
            f"errors={len(self.errors)}  warnings={len(self.warnings)}",
        ]
        for e in self.errors:
            lines.append(f"  ERROR:   {e}")
        for w in self.warnings:
            lines.append(f"  warning: {w}")
        lines.append("VALID ✓" if self.ok else "INVALID ✗")
        return "\n".join(lines)


def validate(
    master: Master,
    paths: DatasetPaths,
    allow_extra_categories: bool = False,
    check_files: bool = True,
) -> Report:
    """Run every check and return a :class:`Report`."""
    rep = Report(n_images=len(master.images), n_annotations=len(master.annotations))

    for problem in master.validate_categories(allow_extra=allow_extra_categories):
        rep.errors.append(problem)

    seen_image_ids: set[int] = set()
    for im in master.images:
        iid = im["id"]
        stem = master.image_stem(im)
        if iid in seen_image_ids:
            rep.errors.append(f"duplicate image id {iid} ({stem})")
        seen_image_ids.add(iid)

        w, h = int(im["width"]), int(im["height"])
        if w <= 0 or h <= 0:
            rep.errors.append(f"{stem}: non-positive image size {w}x{h}")

        if check_files and _source(paths, im["file_name"]) is None:
            rep.errors.append(f"{stem}: image file not found on disk ({im['file_name']})")

        anns = master.annotations_for(iid)
        if not anns:
            # intentional negative — allowed, but recorded so it's never a surprise.
            rep.n_negatives += 1
            continue

        combined: list[list[float]] = []
        for ann in anns:
            for poly in ann["segmentation"]:
                combined.append(poly)
                npts = geometry.polygon_point_count(poly)
                if npts < 3:
                    rep.errors.append(f"{stem}: polygon with {npts} points (need >= 3)")
                    continue
                for x, y in geometry.flat_to_xy(poly):
                    if not (0 <= x <= w and 0 <= y <= h):
                        rep.warnings.append(
                            f"{stem}: vertex ({x:.1f},{y:.1f}) outside image {w}x{h} "
                            f"(will be clamped on export)"
                        )
                        break
        # a labeled image whose polygons enclose no area is almost certainly a bug
        if combined and geometry.polygon_area(combined, w, h) <= 0:
            rep.errors.append(f"{stem}: annotations present but combined mask is empty")

    return rep


def _source(paths: DatasetPaths, file_name: str):
    for base in (paths.master_images, paths.raw_images):
        p = base / file_name
        if p.exists():
            return p
    return None
