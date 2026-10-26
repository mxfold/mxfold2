#!/usr/bin/env python3
"""Prepare manifests for the measured-SHAPE IMLE sweeps.

The historical RibonanzaNet files contain predictions for the full construct,
whereas ``SN0/SHAPE`` contains the experimentally measured 2A3-MaP core after
removing the 26-nt 5' and 21-nt 3' constant regions.  This helper constructs a
small, immutable manifest that makes the distinction explicit and validates
that the measured SHAPE sequence is exactly the corresponding FASTA slice.
It does not copy the large SHAPE files; jobs read them from ``shape_root``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


DEFAULT_DATA = Path("/gs/bs/tga-satolab-gtex/yamauchi/data")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_in(directory: Path, suffix: str) -> set[str]:
    return {
        p.name[: -len(suffix)]
        for p in directory.iterdir()
        if p.is_file() and p.name.endswith(suffix)
    }


def ids_from_list(path: Path) -> set[str]:
    """Read one ID per line, accepting either bare IDs or file paths."""
    ids: set[str] = set()
    with path.open() as handle:
        for line in handle:
            token = line.strip().split()
            if not token or token[0].startswith("#"):
                continue
            name = Path(token[0]).name
            for suffix in (".shape.tsv", ".bpseq"):
                if name.endswith(suffix):
                    name = name[: -len(suffix)]
                    break
            ids.add(name)
    return ids


def read_fasta(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    name: str | None = None
    sequence: list[str] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(sequence)))
                name = line[1:].split()[0]
                sequence = []
            else:
                if name is None:
                    raise ValueError(f"FASTA sequence precedes header in {path}")
                sequence.append(line.upper().replace("T", "U"))
    if name is not None:
        records.append((name, "".join(sequence)))
    return records


def read_shape_sequence_and_values(path: Path) -> tuple[str, list[float]]:
    bases: list[str] = []
    values: list[float] = []
    with path.open() as handle:
        for line in handle:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            bases.append(fields[1].upper().replace("T", "U"))
            try:
                values.append(float(fields[2]))
            except ValueError:
                values.append(float("nan"))
    return "".join(bases), values


def write_lines(path: Path, values: list[str]) -> None:
    path.write_text("".join(f"{value}\n" for value in values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape-root", type=Path, default=DEFAULT_DATA / "Ribonanza/SN0/SHAPE")
    parser.add_argument("--shape-id-list", type=Path, default=None)
    parser.add_argument(
        "--rnafold-root",
        type=Path,
        default=DEFAULT_DATA / "Ribonanza/SN0/PredStructure/RNAfold/BPSEQ",
    )
    parser.add_argument("--rnafold-id-list", type=Path, default=None)
    parser.add_argument(
        "--assisted-root",
        type=Path,
        default=DEFAULT_DATA / "Ribonanza/SN0/PredStructure/Vienna/BPSEQ",
    )
    parser.add_argument("--assisted-id-list", type=Path, default=None)
    parser.add_argument(
        "--cd80-fasta",
        type=Path,
        default=DEFAULT_DATA / "Ribonanza/FASTA/CD80-redundant-u175nt.fasta",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--common-samples", type=int, default=512)
    parser.add_argument("--trim-5", type=int, default=26)
    parser.add_argument("--trim-3", type=int, default=21)
    args = parser.parse_args()

    for path in (args.shape_root, args.rnafold_root, args.assisted_root, args.cd80_fasta):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.shape_id_list, args.rnafold_id_list, args.assisted_id_list):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if args.common_samples < 1:
        raise ValueError("--common-samples must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    shape_ids = (
        ids_from_list(args.shape_id_list)
        if args.shape_id_list is not None
        else ids_in(args.shape_root, ".shape.tsv")
    )
    rnafold_ids = (
        ids_from_list(args.rnafold_id_list)
        if args.rnafold_id_list is not None
        else ids_in(args.rnafold_root, ".bpseq")
    )
    assisted_ids = (
        ids_from_list(args.assisted_id_list)
        if args.assisted_id_list is not None
        else ids_in(args.assisted_root, ".bpseq")
    )
    common_ids = sorted(shape_ids & rnafold_ids & assisted_ids)
    if len(common_ids) < args.common_samples:
        raise RuntimeError(
            f"only {len(common_ids)} measured/RNAfold/assisted IDs, "
            f"cannot select {args.common_samples}"
        )
    common_ids = common_ids[: args.common_samples]

    cd80_records = read_fasta(args.cd80_fasta)
    if len(cd80_records) != 1842:
        raise RuntimeError(f"expected 1842 CD80 records, found {len(cd80_records)}")
    cd80_ids = [name for name, _ in cd80_records]
    if len(set(cd80_ids)) != len(cd80_ids):
        raise RuntimeError("CD80 FASTA contains duplicate IDs")

    # Validate the measured files and construct a core-length CD80 FASTA.  The
    # generated sequence is used for correlation evaluation, so every output
    # position has the same indexing as SN0/SHAPE and SN0/BPSEQ.
    measured_by_id = dict(cd80_records)
    out_fasta = args.out_dir / "cd80_measured_core.fasta"
    stats_values: list[float] = []
    missing_cd80: list[str] = []
    mismatches: list[dict[str, object]] = []
    with out_fasta.open("w") as handle:
        for name, full_sequence in cd80_records:
            shape_path = args.shape_root / f"{name}.shape.tsv"
            if not shape_path.is_file():
                missing_cd80.append(name)
                continue
            shape_sequence, values = read_shape_sequence_and_values(shape_path)
            stop = len(full_sequence) - args.trim_3 if args.trim_3 else len(full_sequence)
            if stop <= args.trim_5:
                raise RuntimeError(f"invalid trimming for {name}: length={len(full_sequence)}")
            expected = full_sequence[args.trim_5 : stop]
            if expected != shape_sequence:
                mismatches.append(
                    {
                        "id": name,
                        "full_length": len(full_sequence),
                        "shape_length": len(shape_sequence),
                        "expected_prefix": expected[:20],
                        "shape_prefix": shape_sequence[:20],
                    }
                )
                continue
            handle.write(f">{name}\n{shape_sequence}\n")
            stats_values.extend(v for v in values if math.isfinite(v) and v > -10.0)

    if missing_cd80:
        raise RuntimeError(f"missing measured SHAPE for {len(missing_cd80)} CD80 IDs")
    if mismatches:
        raise RuntimeError(f"measured/FASTA alignment mismatches: {mismatches[:3]}")

    write_lines(args.out_dir / "common_ids.lst", common_ids)
    write_lines(args.out_dir / "cd80_ids.lst", cd80_ids)
    write_lines(args.out_dir / "shape_ids.lst", sorted(shape_ids))

    # Validate the exact 512 training files as well, and record enough source
    # information to make a scheduler result self-auditing.
    common_lengths: list[int] = []
    for name in common_ids:
        shape_sequence, values = read_shape_sequence_and_values(
            args.shape_root / f"{name}.shape.tsv"
        )
        if not shape_sequence or len(shape_sequence) != len(values):
            raise RuntimeError(f"invalid measured SHAPE file: {name}")
        common_lengths.append(len(shape_sequence))

    manifest = {
        "measurement_source": "Ribonanza SN0 train_data.csv, experiment_type=2A3_MaP",
        "measurement_directory": str(args.shape_root),
        "shape_id_list": str(args.shape_id_list) if args.shape_id_list else None,
        "shape_format": "position, nucleotide, measured reactivity; -999 denotes missing",
        "trimmed_constant_regions": {"five_prime": args.trim_5, "three_prime": args.trim_3},
        "rnafold_directory": str(args.rnafold_root),
        "rnafold_id_list": str(args.rnafold_id_list) if args.rnafold_id_list else None,
        "assisted_directory": str(args.assisted_root),
        "assisted_id_list": str(args.assisted_id_list) if args.assisted_id_list else None,
        "input_cd80_fasta": str(args.cd80_fasta),
        "shape_file_count": len(shape_ids),
        "rnafold_file_count": len(rnafold_ids),
        "assisted_file_count": len(assisted_ids),
        "common_intersection_count": len(shape_ids & rnafold_ids & assisted_ids),
        "common_training_count": len(common_ids),
        "cd80_count": len(cd80_records),
        "cd80_output_fasta_count": sum(1 for line in out_fasta.open() if line.startswith(">")),
        "common_length_min": min(common_lengths),
        "common_length_max": max(common_lengths),
        "common_length_mean": mean(common_lengths),
        "cd80_valid_measurement_positions": len(stats_values),
        "cd80_measurement_min": min(stats_values),
        "cd80_measurement_max": max(stats_values),
        "cd80_measurement_mean": mean(stats_values),
        "cd80_measurement_negative_fraction": sum(v < 0 for v in stats_values) / len(stats_values),
        "files": {
            name: {"sha256": sha256_file(args.out_dir / name), "bytes": (args.out_dir / name).stat().st_size}
            for name in ("common_ids.lst", "cd80_ids.lst", "shape_ids.lst", "cd80_measured_core.fasta")
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
