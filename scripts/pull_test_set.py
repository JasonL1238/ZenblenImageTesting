"""Pull a FRESH pipeline-test image set from the Zenblen Files API.

DB-SAFE: unlike ``labeling/download.py`` this NEVER touches ``labeling/labels.db``
or ``labeling/data/`` — it only reads the DB (to exclude) and writes images into
an isolated folder. This exists so we can run full-pipeline tests on images that
are NOT part of the labeling pipeline.

Exclusion (a pulled image is kept ONLY if it is in NONE of these):
  1. every ``file_id`` and every ``file_name`` in ``labeling/labels.db`` (all 478
     accepted/rejected/skipped/corrected/pending images).
  2. every ``UserGrab_*.jpg`` filename already sorted under ``data/images/`` (the
     92-image set we've been using).

Downloads into ``outputs/pipeline_test_set/images/<file_name>`` (original UserGrab
name → human-readable, matches the 92-set naming and validate_chunks stems) and
writes a manifest CSV. Auth: ``ZENBLEN_API_KEY`` (env or repo .env), never hardcoded.
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://internal-api.zenblen.net/files"
LABELS_DB = ROOT / "labeling" / "labels.db"
SORTED_92_DIRS = [ROOT / "data" / "images" / "red_pink", ROOT / "data" / "images" / "yellow"]
OUT_DIR = ROOT / "outputs" / "pipeline_test_set"
IMG_DIR = OUT_DIR / "images"
MANIFEST = OUT_DIR / "manifest.csv"


def _load_env_key() -> str | None:
    if os.environ.get("ZENBLEN_API_KEY"):
        return os.environ["ZENBLEN_API_KEY"]
    for env_path in (ROOT / "labeling" / ".env", ROOT / ".env"):
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("ZENBLEN_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def build_exclusion() -> tuple[set[int], set[str]]:
    """Return (excluded_file_ids, excluded_file_names) from labels.db + the 92-set."""
    ex_ids: set[int] = set()
    ex_names: set[str] = set()
    if LABELS_DB.exists():
        conn = sqlite3.connect(f"file:{LABELS_DB}?mode=ro", uri=True)
        try:
            for fid, fname in conn.execute("SELECT file_id, file_name FROM files"):
                if fid is not None:
                    ex_ids.add(int(fid))
                if fname:
                    ex_names.add(str(fname))
        finally:
            conn.close()
    for d in SORTED_92_DIRS:
        if d.exists():
            for p in d.glob("*.jpg"):
                ex_names.add(p.name)
    return ex_ids, ex_names


def fetch_file_list(api_key: str, start: str, end: str, category: str | None,
                    file_type: str | None, timeout: float) -> list[dict]:
    body: dict[str, object] = {"startTime": start, "endTime": end}
    if category:
        body["fileCategoryName"] = category
    if file_type:
        body["fileTypeName"] = file_type
    resp = requests.post(API_URL,
                         headers={"Content-Type": "application/json", "x-api-key": api_key},
                         json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array, got {type(data).__name__}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Pull a fresh, labeling-disjoint test set")
    ap.add_argument("--start", required=True, help="'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--end", required=True, help="'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--category", default="UserGrab", help="fileCategoryName (default UserGrab)")
    ap.add_argument("--type", dest="file_type", default="image/jpg")
    ap.add_argument("--limit", type=int, default=0, help="cap #downloads (0 = no cap)")
    ap.add_argument("--list-only", action="store_true", help="report counts, download nothing")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    api_key = _load_env_key()
    if not api_key:
        ap.error("no API key: set ZENBLEN_API_KEY (env or .env)")

    ex_ids, ex_names = build_exclusion()
    print(f"exclusion set: {len(ex_ids)} file_ids, {len(ex_names)} file_names")

    print(f"fetching {args.start} .. {args.end} (category={args.category}, type={args.file_type})")
    files = fetch_file_list(api_key, args.start, args.end, args.category,
                            args.file_type or None, args.timeout)
    print(f"  API returned {len(files)} files")

    # keep only files disjoint from BOTH exclusion keys; also require a url + name
    kept: list[dict] = []
    seen_names: set[str] = set()
    for f in files:
        fid = f.get("file_id")
        fname = f.get("file_name", "") or ""
        url = f.get("file_url", "") or ""
        if not fname or not url:
            continue
        if fid is not None and int(fid) in ex_ids:
            continue
        if fname in ex_names:
            continue
        if fname in seen_names:  # de-dup within the API response
            continue
        seen_names.add(fname)
        kept.append(f)

    print(f"  {len(kept)} kept after excluding labeling-pipeline + 92-set images")
    if args.limit and len(kept) > args.limit:
        # spread across the window: keep every Nth by created_at order
        kept.sort(key=lambda f: f.get("created_at", ""))
        step = len(kept) / args.limit
        kept = [kept[int(i * step)] for i in range(args.limit)]
        print(f"  capped to {len(kept)} (evenly sampled across the window)")

    if args.list_only:
        print("--list-only: nothing downloaded")
        # still emit a manifest of the candidate set for inspection
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    dl = skip = err = 0
    with requests.Session() as s:
        for i, f in enumerate(kept, 1):
            fname = f["file_name"]
            dest = IMG_DIR / fname
            rows.append({"file_id": f.get("file_id"), "file_name": fname,
                         "created_at": f.get("created_at", ""),
                         "category_name": f.get("category_name", "")})
            if args.list_only:
                continue
            if dest.exists():
                skip += 1
                continue
            try:
                r = s.get(f["file_url"], timeout=args.timeout)
                r.raise_for_status()
                dest.write_bytes(r.content)
                dl += 1
                if i % 25 == 0:
                    print(f"  [{i}/{len(kept)}] downloaded")
            except requests.RequestException as e:
                err += 1
                print(f"  [{i}/{len(kept)}] {fname} ERROR {e}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_id", "file_name", "created_at", "category_name"])
        w.writeheader()
        w.writerows(rows)
    print(f"manifest: {MANIFEST} ({len(rows)} candidates)")
    if not args.list_only:
        print(f"done: {dl} downloaded, {skip} already on disk, {err} errors -> {IMG_DIR}")


if __name__ == "__main__":
    main()
