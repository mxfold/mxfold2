#!/usr/bin/env python3
"""Convert one temporary RNAgym BPP directory to a compact shard archive."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rnagym.evaluate_rnagym import (  # noqa: E402
    MODIFIERS,
    parse_reactivity,
    read_prediction_map,
    unpaired_from_bpp,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--shard-map", type=Path, required=True)
    parser.add_argument("--bpp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.test_parquet,
                            columns=["sequence", "modifier", "SNR", "reactivity"])
    frame = frame[frame["modifier"].isin(MODIFIERS)].copy()
    frame["sequence"] = (frame["sequence"].astype(str).str.upper()
                          .str.replace("T", "U", regex=False))
    frame = (frame.sort_values(["modifier", "sequence", "SNR"],
                               ascending=[True, True, False], kind="mergesort")
                  .drop_duplicates(["modifier", "sequence"], keep="first"))
    prediction_map = read_prediction_map(args.shard_map)
    frame = frame[frame["sequence"].isin(prediction_map)].copy()

    arrays: dict[str, np.ndarray] = {}
    summary: dict[str, object] = {"missing_sequences": {}, "n_sequences": {}}
    for modifier in MODIFIERS:
        true_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        missing = 0
        modifier_frame = frame[frame["modifier"] == modifier]
        for row in modifier_frame.itertuples(index=False):
            sequence = str(row.sequence)
            prediction_id = prediction_map[sequence]
            bpp_path = args.bpp_dir / f"{prediction_id}.bpp"
            if not bpp_path.exists():
                missing += 1
                continue
            true_parts.append(parse_reactivity(row.reactivity, len(sequence)))
            pred_parts.append(unpaired_from_bpp(bpp_path, len(sequence)))
        if missing:
            raise FileNotFoundError(f"{modifier}: {missing} predictions missing in shard")
        arrays[f"{modifier}_true"] = (np.concatenate(true_parts) if true_parts
                                       else np.empty(0, dtype=np.float64))
        arrays[f"{modifier}_pred"] = (np.concatenate(pred_parts) if pred_parts
                                       else np.empty(0, dtype=np.float64))
        arrays[f"{modifier}_n_sequences"] = np.asarray([len(modifier_frame)], dtype=np.int64)
        summary["missing_sequences"][modifier] = missing  # type: ignore[index]
        summary["n_sequences"][modifier] = len(modifier_frame)  # type: ignore[index]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, args.out)
    summary["n_rows"] = int(len(frame))
    summary["out"] = str(args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
