#!/usr/bin/env python3
"""Create the small amount of list metadata needed by the REPORT benchmark.

The expensive RNAgym preparation is intentionally not repeated here.  The
existing per-modifier RNAfold lists already contain cached BPSEQ files; this
script only makes one deduplicated-by-path list for the single RNAfold-only
condition and records the source lists.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def read_entries(path: Path) -> list[str]:
    entries: list[str] = []
    with path.open() as handle:
        for line in handle:
            value = line.strip()
            if value:
                entries.append(value)
    return entries


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dms-list", type=Path, required=True)
    parser.add_argument("--shape-list", type=Path, required=True,
                        help="2A3 list; RNAgym's SHAPE-like modifier")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dms = read_entries(args.dms_list)
    shape = read_entries(args.shape_list)
    combined: list[str] = []
    seen: set[str] = set()
    for entry in (*dms, *shape):
        if entry not in seen:
            seen.add(entry)
            combined.append(entry)

    args.out.mkdir(parents=True, exist_ok=True)
    atomic_write(args.out / "rnafold.lst", "".join(f"{entry}\n" for entry in combined))
    manifest = {
        "rnafold_list": str((args.out / "rnafold.lst").resolve()),
        "source_dms_list": str(args.dms_list.resolve()),
        "source_shape_2a3_list": str(args.shape_list.resolve()),
        "source_entries": {"DMS": len(dms), "2A3": len(shape)},
        "combined_entries": len(combined),
        "deduplication": "exact BPSEQ path",
        "shape_label": "2A3",
    }
    atomic_write(args.out / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
