#!/usr/bin/env python3
"""
Build a global deduped gentag table: one row per distinct normalized tag.

Reads Phase 1 per-line tag CSVs (*_tags_<model>.csv), keeps rows with
status=success, aggregates tag_norm (lower/strip) across all venues and models.

Output columns:
  tag_id, tag, frequency, modelCount, wordCount, models, prompts, extraction_run_id

tag_id is UUID v5(TAG_ID_NAMESPACE, tag) so the same normalized tag always gets
the same id when this script is re-run.

Usage:
  python scripts/export_gentags_global_deduped.py

Writes data/gentags_global_deduped.csv by default (override with --output).
"""

from __future__ import annotations

import argparse
import csv
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from glob import glob

# Stable namespace for tag string -> tag_id. Do not change between exports.
TAG_ID_NAMESPACE = uuid.UUID("f47ac10b-58cc-4372-a567-0e02b2c3d479")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    default_glob = str(root / "results/phase1_downloaded/week2_run_*_tags_*.csv")
    parser = argparse.ArgumentParser(description="Export global deduped gentags CSV.")
    parser.add_argument(
        "--tags-glob",
        type=str,
        default=default_glob,
        help="Glob for per-line tag CSV files (default: phase1_downloaded/week2_run_*_tags_*.csv).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(root / "data/gentags_global_deduped.csv"),
        help="Output CSV path (default: data/gentags_global_deduped.csv).",
    )
    parser.add_argument(
        "--extraction-run-id",
        type=str,
        default="",
        help="Stored in extraction_run_id column. Default: inferred from first input filename.",
    )
    return parser.parse_args()


def infer_run_id(paths: list[Path]) -> str:
    """e.g. week2_run_20251223_191104 from week2_run_20251223_191104_tags_openai.csv."""
    if not paths:
        return "unknown"
    name = paths[0].name
    if "_tags_" in name:
        return name.split("_tags_")[0]
    return paths[0].stem


def main() -> int:
    args = parse_args()
    paths = sorted(Path(p) for p in glob(args.tags_glob))
    if not paths:
        print(f"No files matched: {args.tags_glob}", file=sys.stderr)
        return 1

    run_id = args.extraction_run_id or infer_run_id(paths)
    out_path = Path(args.output)

    # tag -> aggregates
    agg: dict[str, dict] = defaultdict(
        lambda: {
            "frequency": 0,
            "models": set(),
            "prompts": set(),
            "word_counts": [],
        }
    )

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
                bucket = agg[tag]
                bucket["frequency"] += 1
                mk = (row.get("model_key") or "").strip()
                if mk:
                    bucket["models"].add(mk)
                pt = (row.get("prompt_type") or "").strip()
                if pt:
                    bucket["prompts"].add(pt)
                try:
                    bucket["word_counts"].append(int(row.get("word_count") or 0))
                except ValueError:
                    bucket["word_counts"].append(0)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "tag_id",
        "tag",
        "frequency",
        "modelCount",
        "wordCount",
        "models",
        "prompts",
        "extraction_run_id",
    ]

    rows_out = []
    for tag, data in agg.items():
        wcs = data["word_counts"]
        word_count = max(wcs) if wcs else 0
        models = ",".join(sorted(data["models"]))
        prompts = ",".join(sorted(data["prompts"]))
        tid = str(uuid.uuid5(TAG_ID_NAMESPACE, tag))
        rows_out.append(
            {
                "tag_id": tid,
                "tag": tag,
                "frequency": data["frequency"],
                "modelCount": len(data["models"]),
                "wordCount": word_count,
                "models": models,
                "prompts": prompts,
                "extraction_run_id": run_id,
            }
        )

    rows_out.sort(key=lambda r: (-r["frequency"], r["tag"]))

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"Wrote {len(rows_out)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
