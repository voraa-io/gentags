#!/usr/bin/env python3
"""
Export deduped gentags: one row per (venue_id, tag) from Phase 1 *_tags_<model>.csv.

Uses status=success, tag_norm strip+lower. tag_id is UUID v5(VENUE_TAG_ID_NAMESPACE,
f"{venue_id}\\0{tag}").

Usage:
  python scripts/export_gentags_deduped_by_venue.py

Default output: data/gentags_deduped_by_venue.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from glob import glob

VENUE_TAG_ID_NAMESPACE = uuid.UUID("a8b3c4d5-e6f7-4890-a1b2-c3d4e5f60789")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_glob = str(root / "results/phase1_downloaded/week2_run_*_tags_*.csv")
    parser = argparse.ArgumentParser(description="Export gentags deduped by venue.")
    parser.add_argument(
        "--tags-glob",
        type=str,
        default=default_glob,
        help="Glob for per-line tag CSV files (default: phase1_downloaded/week2_run_*_tags_*.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(root / "data/gentags_deduped_by_venue.csv"),
        help="Output CSV (default: data/gentags_deduped_by_venue.csv).",
    )
    parser.add_argument(
        "--extraction-run-id",
        type=str,
        default="",
        help="Stored in extraction_run_id column. Default: inferred from first input filename.",
    )
    return parser.parse_args()


def infer_run_id(paths: list[Path]) -> str:
    if not paths:
        return "unknown"
    name = paths[0].name
    if "_tags_" in name:
        return name.split("_tags_")[0]
    return paths[0].stem


def empty_bucket() -> dict:
    return {
        "frequency": 0,
        "models": set(),
        "prompts": set(),
        "word_counts": [],
    }


def main() -> int:
    args = parse_args()
    paths = sorted(Path(p) for p in glob(args.tags_glob))
    if not paths:
        print(f"No files matched: {args.tags_glob}", file=sys.stderr)
        return 1

    run_id = args.extraction_run_id or infer_run_id(paths)
    out_path = Path(args.output)

    agg_venue: dict[tuple[str, str], dict] = defaultdict(empty_bucket)
    venue_names: dict[str, str] = {}

    for fp in paths:
        with fp.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("status") != "success":
                    continue
                raw = row.get("tag_norm") or ""
                tag = raw.strip().lower()
                if not tag:
                    continue
                vid = (row.get("venue_id") or "").strip()
                if not vid:
                    continue
                vname = (row.get("venue_name") or "").strip()
                venue_names[vid] = vname

                mk = (row.get("model_key") or "").strip()
                pt = (row.get("prompt_type") or "").strip()
                key = (vid, tag)
                v = agg_venue[key]
                v["frequency"] += 1
                if mk:
                    v["models"].add(mk)
                if pt:
                    v["prompts"].add(pt)
                try:
                    v["word_counts"].append(int(row.get("word_count") or 0))
                except ValueError:
                    v["word_counts"].append(0)

    def finalize_row(data: dict) -> dict:
        wcs = data["word_counts"]
        word_count = max(wcs) if wcs else 0
        return {
            "frequency": data["frequency"],
            "modelCount": len(data["models"]),
            "wordCount": word_count,
            "models": ",".join(sorted(data["models"])),
            "prompts": ",".join(sorted(data["prompts"])),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "venue_id",
        "venueName",
        "tag_id",
        "tag",
        "frequency",
        "modelCount",
        "wordCount",
        "models",
        "prompts",
        "extraction_run_id",
    ]
    rows_v = []
    for (vid, tag), data in agg_venue.items():
        fin = finalize_row(data)
        pair_key = f"{vid}\0{tag}"
        rows_v.append(
            {
                "venue_id": vid,
                "venueName": venue_names.get(vid, ""),
                "tag_id": str(uuid.uuid5(VENUE_TAG_ID_NAMESPACE, pair_key)),
                "tag": tag,
                "frequency": fin["frequency"],
                "modelCount": fin["modelCount"],
                "wordCount": fin["wordCount"],
                "models": fin["models"],
                "prompts": fin["prompts"],
                "extraction_run_id": run_id,
            }
        )
    rows_v.sort(key=lambda r: (r["venue_id"], -r["frequency"], r["tag"]))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_v)
    print(f"Wrote {len(rows_v)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
