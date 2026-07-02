"""Diff two chunk-report scores.csv files: verdict flips + big pixel-count moves.

Usage: /opt/miniconda3/bin/python scripts/diff_reports.py <a/scores.csv> <b/scores.csv>
"""
from __future__ import annotations

import csv
import sys


def load(p):
    with open(p) as f:
        return {r["stem"]: r for r in csv.DictReader(f)}


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    flips, moves = [], []
    for stem in sorted(a):
        ra, rb = a[stem], b.get(stem)
        if rb is None:
            continue
        pa, pb = int(ra["chunk_pixels"]), int(rb["chunk_pixels"])
        if ra["verdict"] != rb["verdict"]:
            flips.append((stem, ra["verdict"], rb["verdict"], pa, pb))
        elif abs(pb - pa) > 300:
            moves.append((stem, ra["verdict"], pa, pb))
    print(f"verdict flips: {len(flips)}")
    for s, va, vb, pa, pb in flips:
        print(f"  {s[:40]}  {va}->{vb}  px {pa}->{pb}")
    print(f"same-verdict big pixel moves (>300px): {len(moves)}")
    for s, v, pa, pb in moves:
        print(f"  {s[:40]}  [{v}]  px {pa}->{pb}")


if __name__ == "__main__":
    main()
