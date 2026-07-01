"""Geometry conversions shared by every exporter.

A *polygon* here is a flat list of alternating pixel coordinates
``[x1, y1, x2, y2, ...]`` — the COCO segmentation convention. A COCO annotation's
``segmentation`` field is a *list* of such polygons (one instance may be split
into several disjoint pieces), so most helpers accept ``list[list[float]]``.

All conversions are pure functions (no I/O) so they are trivially unit-testable.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def flat_to_xy(flat: Sequence[float]) -> list[tuple[float, float]]:
    """``[x1, y1, x2, y2, ...] -> [(x1, y1), (x2, y2), ...]``."""
    if len(flat) % 2 != 0:
        raise ValueError(f"polygon has odd coordinate count ({len(flat)})")
    return [(float(flat[i]), float(flat[i + 1])) for i in range(0, len(flat), 2)]


def polygon_point_count(flat: Sequence[float]) -> int:
    """Number of (x, y) vertices in a flat polygon."""
    return len(flat) // 2


def polygons_to_mask(
    polygons: Sequence[Sequence[float]], width: int, height: int, fg: int = 1
) -> np.ndarray:
    """Rasterize one or more flat polygons into a single-channel mask.

    Pixels inside any polygon are set to ``fg``; everything else stays 0. Polygons
    with fewer than 3 points are skipped (they enclose no area).
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    cv_polys: list[np.ndarray] = []
    for poly in polygons:
        if polygon_point_count(poly) < 3:
            continue
        pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
        cv_polys.append(np.round(pts).astype(np.int32))
    if cv_polys:
        cv2.fillPoly(mask, cv_polys, color=int(fg))
    return mask


def mask_to_polygons(
    mask: np.ndarray, epsilon_frac: float = 0.0, min_points: int = 3
) -> list[list[float]]:
    """Extract external contours of a binary mask as flat polygons.

    ``epsilon_frac`` > 0 simplifies each contour with ``approxPolyDP`` (tolerance
    as a fraction of the contour perimeter). Contours that reduce to fewer than
    ``min_points`` vertices are dropped.
    """
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) <= 0:
            continue
        if epsilon_frac > 0:
            eps = epsilon_frac * cv2.arcLength(cnt, True)
            cnt = cv2.approxPolyDP(cnt, eps, True)
        pts = cnt.reshape(-1, 2)
        if len(pts) < min_points:
            continue
        polygons.append([float(v) for xy in pts for v in xy])
    return polygons


def normalize_polygon(flat: Sequence[float], width: int, height: int) -> list[float]:
    """Scale pixel coordinates into the ``[0, 1]`` range (YOLO convention).

    Values are clamped to ``[0, 1]`` so a vertex sitting exactly on the image edge
    (or a pixel outside from a sloppy annotation) never emits an out-of-range label.
    """
    out: list[float] = []
    for i, v in enumerate(flat):
        denom = width if i % 2 == 0 else height
        out.append(min(1.0, max(0.0, float(v) / denom)))
    return out


def polygon_bbox(polygons: Sequence[Sequence[float]]) -> list[float]:
    """Axis-aligned ``[x, y, w, h]`` bounding box over all given polygons."""
    xs: list[float] = []
    ys: list[float] = []
    for poly in polygons:
        for x, y in flat_to_xy(poly):
            xs.append(x)
            ys.append(y)
    if not xs:
        return [0.0, 0.0, 0.0, 0.0]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def polygon_area(polygons: Sequence[Sequence[float]], width: int, height: int) -> float:
    """Area in pixels of the union of the polygons (rasterized mask pixel count).

    Using the rasterized count (rather than the shoelace sum) makes the value
    match what COCO consumers expect for polygon segmentation and correctly
    handles overlapping / self-touching parts.
    """
    mask = polygons_to_mask(polygons, width, height, fg=1)
    return float(int(mask.sum()))


def to_yolo_rows(
    polygons_per_object: Sequence[Sequence[Sequence[float]]],
    width: int,
    height: int,
    class_id: int = 0,
) -> list[str]:
    """Build YOLO-seg label rows, one per polygon.

    ``polygons_per_object`` is a list of annotations, each a list of flat polygons.
    Ultralytics treats each row as one instance made of a single polygon ring, so a
    multi-part annotation is emitted as one row *per part*. Coordinates are
    normalized to ``[0, 1]``; the ring is not explicitly closed (YOLO closes it).
    """
    rows: list[str] = []
    for obj in polygons_per_object:
        for poly in obj:
            if polygon_point_count(poly) < 3:
                continue
            norm = normalize_polygon(poly, width, height)
            coords = " ".join(f"{c:.6f}" for c in norm)
            rows.append(f"{class_id} {coords}")
    return rows
