from __future__ import annotations

import csv
import math
from argparse import ArgumentParser
import sys


MISSING_TOKENS = {"na", "nan", ""}


def is_bpseq_format(path: str) -> bool:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                return False
            try:
                int(fields[0])
                int(fields[2])
                return True
            except ValueError:
                return False
    return False


def is_shape_format(path: str) -> bool:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                return False
            try:
                int(fields[0])
                if fields[1].lower() not in MISSING_TOKENS:
                    float(fields[1])
                return True
            except ValueError:
                return False
    return False


def read_list_file(path: str) -> list[str]:
    files: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            files.append(line)
    return files


def read_bpseq_pairs(path: str) -> list[int]:
    pairs = [0]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(f"invalid BPSEQ line in {path}: {line}")
            idx = int(fields[0])
            pair = int(fields[2])
            while len(pairs) <= idx:
                pairs.append(0)
            pairs[idx] = pair
    return pairs


def parse_reactivity_value(token: str) -> float | None:
    t = token.strip()
    if t.lower() in MISSING_TOKENS:
        return None
    v = float(t)
    if math.isnan(v) or v < 0:
        return None
    return v


def read_shape_reactivity(path: str) -> dict[int, float]:
    values: dict[int, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"invalid SHAPE line in {path}: {line}")
            pos = int(fields[0])
            value = parse_reactivity_value(fields[1])
            if value is not None:
                values[pos] = value
    return values


def pairwise_consistency(
    pairs: list[int], shape: dict[int, float]
) -> tuple[float, int, int]:
    unpaired_scores: list[float] = []
    paired_scores: list[float] = []

    for i in range(1, len(pairs)):
        if i not in shape:
            continue
        if pairs[i] == 0:
            unpaired_scores.append(shape[i])
        else:
            paired_scores.append(shape[i])

    compared = len(unpaired_scores) * len(paired_scores)
    if compared == 0:
        return float("nan"), len(unpaired_scores) + len(paired_scores), 0

    consistent = 0.0
    for u in unpaired_scores:
        for p in paired_scores:
            if u > p:
                consistent += 1.0
            elif u == p:
                consistent += 0.5

    return consistent / compared, len(unpaired_scores) + len(paired_scores), compared


def main() -> None:
    parser = ArgumentParser(
        description=(
            "Compare predicted BPSEQ structures and SHAPE reactivity using "
            "pairwise consistency (AUROC-like): P(reactivity(unpaired) > reactivity(paired))."
        ),
        add_help=True,
    )
    parser.add_argument("bpseq", help="BPSEQ file or list of BPSEQ file paths")
    parser.add_argument(
        "shape", help="SHAPE reactivity file or list of SHAPE file paths"
    )
    parser.add_argument(
        "--output",
        choices=["summary", "detailed", "csv"],
        default="summary",
        help="output format (default: summary)",
    )
    args = parser.parse_args()

    if is_bpseq_format(args.bpseq):
        bpseq_files = [args.bpseq]
    else:
        bpseq_files = read_list_file(args.bpseq)

    if is_shape_format(args.shape):
        shape_files = [args.shape]
    else:
        shape_files = read_list_file(args.shape)

    if len(bpseq_files) != len(shape_files):
        raise ValueError(
            f"list length mismatch: bpseq={len(bpseq_files)}, shape={len(shape_files)}"
        )

    results: list[tuple[str, str, float, int, int]] = []
    for bpseq_path, shape_path in zip(bpseq_files, shape_files):
        pairs = read_bpseq_pairs(bpseq_path)
        shape = read_shape_reactivity(shape_path)
        score, n_valid, n_compared = pairwise_consistency(pairs, shape)
        results.append((bpseq_path, shape_path, score, n_valid, n_compared))

    valid_scores = [r[2] for r in results if not math.isnan(r[2])]
    mean_score = (
        float("nan")
        if len(valid_scores) == 0
        else sum(valid_scores) / len(valid_scores)
    )

    if args.output == "summary":
        print(f"n={len(results)}")
        print(f"n_valid={len(valid_scores)}")
        print(
            f"mean_pairwise_consistency={mean_score:.6f}"
            if not math.isnan(mean_score)
            else "mean_pairwise_consistency=nan"
        )
        return

    if args.output == "detailed":
        print("bpseq\tshape\tpairwise_consistency\tn_valid_positions\tn_compared_pairs")
        for bpseq_path, shape_path, score, n_valid, n_compared in results:
            s = "nan" if math.isnan(score) else f"{score:.6f}"
            print(f"{bpseq_path}\t{shape_path}\t{s}\t{n_valid}\t{n_compared}")
        s = "nan" if math.isnan(mean_score) else f"{mean_score:.6f}"
        print(f"MEAN\t-\t{s}\t-\t-")
        return

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "bpseq",
            "shape",
            "pairwise_consistency",
            "n_valid_positions",
            "n_compared_pairs",
        ]
    )
    for bpseq_path, shape_path, score, n_valid, n_compared in results:
        writer.writerow(
            [
                bpseq_path,
                shape_path,
                "nan" if math.isnan(score) else f"{score:.6f}",
                n_valid,
                n_compared,
            ]
        )
    writer.writerow(
        [
            "MEAN",
            "-",
            "nan" if math.isnan(mean_score) else f"{mean_score:.6f}",
            "-",
            "-",
        ]
    )


if __name__ == "__main__":
    main()
