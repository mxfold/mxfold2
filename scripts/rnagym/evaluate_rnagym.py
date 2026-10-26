#!/usr/bin/env python3
"""Score mxfold2 unpaired probabilities against RNAgym chemical mapping.

The chemical metrics follow the official RNAgym scorer: clipped reactivities
and predictions, MAE, ROC-AUC, MCC, and macro-F1 after independent median
binning.  A BPSEQ result CSV can additionally be supplied for TestSetB
structure metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


MODIFIERS = ("DMS", "2A3")
EVALUATION_VERSION = "symmetric-bpp-v2"


def unpaired_from_edges(edges, length: int) -> np.ndarray:
    """Sum each undirected pair at both endpoints, exactly once.

    Native MXfold2 output is upper triangular. Symmetric input is also
    accepted, without double counting its reverse entries.
    """
    pairs: dict[tuple[int, int], float] = {}
    for left, right, probability in edges:
        if not (1 <= left <= length and 1 <= right <= length):
            raise ValueError(f"BPP indices {(left, right)} outside length {length}")
        if left == right:
            raise ValueError("BPP diagonal entries are not base pairs")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError(f"Invalid BPP probability: {probability}")
        key = (min(left, right), max(left, right))
        if key in pairs and not math.isclose(pairs[key], probability, abs_tol=1e-6):
            raise ValueError(f"Conflicting probabilities for pair {key}")
        pairs[key] = probability
    paired = np.zeros(length, dtype=np.float64)
    for (left, right), probability in pairs.items():
        paired[left - 1] += probability
        paired[right - 1] += probability
    return 1.0 - np.clip(paired, 0.0, 1.0)


def parse_reactivity(value: object, length: int) -> np.ndarray:
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    values = np.fromstring(text, sep=",", dtype=float)
    if values.size != length:
        raise ValueError(f"reactivity length {values.size} != sequence length {length}")
    return values


def unpaired_from_bpp(path: Path, length: int) -> np.ndarray:
    edges = []
    with path.open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) < 3:
                continue
            index = int(fields[0]) - 1
            if not 0 <= index < length:
                raise ValueError(f"index {index + 1} out of range in {path}")
            for pair_probability in fields[2:]:
                if ":" not in pair_probability:
                    continue
                partner_text, probability_text = pair_probability.split(":", 1)
                partner = int(partner_text) - 1
                probability = float(probability_text)
                if not 0 <= partner < length:
                    raise ValueError(f"partner {partner + 1} out of range in {path}")
                edges.append((index + 1, partner + 1, probability))
    return unpaired_from_edges(edges, length)


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    ranks = rankdata(scores, method="average")
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) /
                 (positives.sum() * negatives.sum()))


def mcc(labels: np.ndarray, predictions: np.ndarray) -> float:
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    # Cast before multiplication: the product can exceed int64 for the
    # RNAgym subset, which otherwise turns the numpy sqrt input into an
    # object scalar under recent numpy versions.
    denominator = math.sqrt(float(tp + fp) * float(tp + fn) *
                            float(tn + fp) * float(tn + fn))
    return float((tp * tn - fp * fn) / denominator) if denominator else 0.0


def f1_binary(labels: np.ndarray, predictions: np.ndarray, positive: int) -> float:
    tp = np.sum((labels == positive) & (predictions == positive))
    fp = np.sum((labels != positive) & (predictions == positive))
    fn = np.sum((labels == positive) & (predictions != positive))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def chemical_metrics(true: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(true) & np.isfinite(pred)
    true = np.clip(true[valid], 0.0, 1.0)
    pred = np.clip(pred[valid], 0.0, 1.0)
    true_labels = (true > np.median(true)).astype(np.int8)
    pred_labels = (pred > np.median(pred)).astype(np.int8)
    return {
        "n_bases": int(true.size),
        "true_median": float(np.median(true)),
        "pred_median": float(np.median(pred)),
        "auc": safe_auc(true_labels, pred),
        "mcc": mcc(true_labels, pred_labels),
        "mae": float(np.mean(np.abs(true - pred))),
        "f1": float((f1_binary(true_labels, pred_labels, 1) +
                     f1_binary(true_labels, pred_labels, 0)) / 2),
    }


def read_prediction_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mapping[row["sequence"]] = row["prediction_id"]
    return mapping


def load_structure_metrics(path: Path) -> dict[str, float | int]:
    rows: list[dict[str, str]] = []
    with path.open() as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))
    if not rows:
        raise ValueError(f"no structure rows found in {path}")
    values = {key: np.asarray([float(row[key]) for row in rows], dtype=float)
              for key in ("SEN", "PPV", "F")}
    counts = {key: sum(float(row[key]) for row in rows) for key in ("TP", "TN", "FP", "FN")}
    tp, tn, fp, fn = (counts[key] for key in ("TP", "TN", "FP", "FN"))
    sens_den = tp + fn
    ppv_den = tp + fp
    f1_den = 2 * tp + fp + fn
    return {
        "n_sequences": len(rows),
        "mean_SEN": float(np.nanmean(values["SEN"])),
        "mean_PPV": float(np.nanmean(values["PPV"])),
        "mean_F1": float(np.nanmean(values["F"])),
        "micro_SEN": float(tp / sens_den) if sens_den else 0.0,
        "micro_PPV": float(tp / ppv_den) if ppv_den else 0.0,
        "micro_F1": float(2 * tp / f1_den) if f1_den else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--test-map", type=Path, required=True)
    parser.add_argument("--bpp-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--structure-result", type=Path, default=None)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument(
        "--subset-only",
        action="store_true",
        help="score only sequences present in --test-map (for a test subset)",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.test_parquet, columns=["sequence", "modifier", "SNR", "reactivity"])
    frame = frame[frame["modifier"].isin(MODIFIERS)].copy()
    # RNAgym's sequence-level organization keeps the highest-SNR observation
    # per sequence/modifier before concatenating base-level observations.
    frame["sequence"] = frame["sequence"].astype(str).str.upper().str.replace("T", "U", regex=False)
    frame = (frame.sort_values(["modifier", "sequence", "SNR"], ascending=[True, True, False], kind="mergesort")
                  .drop_duplicates(["modifier", "sequence"], keep="first"))
    prediction_map = read_prediction_map(args.test_map)
    if args.subset_only:
        frame = frame[frame["sequence"].isin(prediction_map)].copy()

    metrics: dict[str, dict[str, float | int]] = {}
    missing: dict[str, int] = {}
    for modifier in MODIFIERS:
        true_parts: list[np.ndarray] = []
        pred_parts: list[np.ndarray] = []
        missing_count = 0
        modifier_frame = frame[frame["modifier"] == modifier]
        for row in modifier_frame.itertuples(index=False):
            sequence = str(row.sequence)
            prediction_id = prediction_map.get(sequence)
            if prediction_id is None:
                missing_count += 1
                continue
            bpp_path = args.bpp_dir / f"{prediction_id}.bpp"
            if not bpp_path.exists():
                missing_count += 1
                continue
            truth = parse_reactivity(row.reactivity, len(sequence))
            prediction = unpaired_from_bpp(bpp_path, len(sequence))
            true_parts.append(truth)
            pred_parts.append(prediction)
        missing[modifier] = missing_count
        if missing_count and not args.allow_missing:
            raise FileNotFoundError(f"{modifier}: {missing_count} sequence predictions are missing")
        if true_parts:
            metrics[modifier] = chemical_metrics(np.concatenate(true_parts), np.concatenate(pred_parts))
        else:
            metrics[modifier] = {"n_bases": 0}
        metrics[modifier]["n_sequences"] = int(len(modifier_frame) - missing_count)
        metrics[modifier]["missing_sequences"] = missing_count

    all_true = []
    all_pred = []
    for modifier in MODIFIERS:
        # Re-read the small set of selected rows so All uses the same valid
        # observations as the per-modifier rows.
        modifier_frame = frame[frame["modifier"] == modifier]
        for row in modifier_frame.itertuples(index=False):
            prediction_id = prediction_map.get(str(row.sequence))
            bpp_path = args.bpp_dir / f"{prediction_id}.bpp" if prediction_id else None
            if bpp_path is None or not bpp_path.exists():
                continue
            all_true.append(parse_reactivity(row.reactivity, len(row.sequence)))
            all_pred.append(unpaired_from_bpp(bpp_path, len(row.sequence)))
    if all_true:
        metrics["All"] = chemical_metrics(np.concatenate(all_true), np.concatenate(all_pred))
    else:
        metrics["All"] = {"n_bases": 0}
    metrics["All"]["n_sequences"] = sum(int(metrics[m].get("n_sequences", 0)) for m in MODIFIERS)
    metrics["All"]["missing_sequences"] = sum(missing.values())

    result: dict[str, object] = {
        "evaluation_version": EVALUATION_VERSION,
        "chemical": metrics,
        "missing_sequences": missing,
        "test_rows_after_sequence_deduplication": int(len(frame)),
    }
    if args.structure_result is not None:
        result["structure"] = load_structure_metrics(args.structure_result)

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "metrics.json").open("w") as handle:
        json.dump(result, handle, indent=2, allow_nan=True)
        handle.write("\n")
    with (args.out / "chemical_metrics.csv").open("w", newline="") as handle:
        fieldnames = ["modifier", "n_sequences", "missing_sequences", "n_bases", "auc", "mcc", "mae", "f1", "true_median", "pred_median"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for modifier in (*MODIFIERS, "All"):
            row = {"modifier": modifier}
            row.update(metrics[modifier])
            writer.writerow(row)
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
