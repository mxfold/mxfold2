#!/usr/bin/env python3
"""Prepare a matched measured-SHAPE/RNet comparison set.

The SN0 measured files are core sequences, while SHAPE-rnet files contain the
same construct with 26 nt at the 5' end and 21 nt at the 3' end.  This helper
selects IDs present in both sources and in both pseudo-label directories, then
writes only the trimmed RNet files needed by the comparison jobs.  Measured
files remain in their original directory; the manifest records all hashes and
paths needed to audit the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def ids_from_list(path: Path) -> set[str]:
    out: set[str] = set()
    for line in path.read_text().splitlines():
        fields = line.strip().split()
        if not fields or fields[0].startswith("#"):
            continue
        name = Path(fields[0]).name
        for suffix in (".shape.tsv", ".bpseq"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        out.add(name)
    return out


def ids_in(root: Path, suffix: str) -> set[str]:
    return {p.name[: -len(suffix)] for p in root.glob(f"*{suffix}") if p.is_file()}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_shape(path: Path) -> tuple[str, list[tuple[str, str]]]:
    bases: list[str] = []
    rows: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 3:
            continue
        bases.append(fields[1].upper().replace("T", "U"))
        rows.append((fields[1], fields[2]))
    return "".join(bases), rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured-root", type=Path, required=True)
    parser.add_argument("--measured-id-list", type=Path, required=True)
    parser.add_argument("--rnet-root", type=Path, required=True)
    parser.add_argument("--rnafold-root", type=Path, required=True)
    parser.add_argument("--assisted-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--trim-5", type=int, default=26)
    parser.add_argument("--trim-3", type=int, default=21)
    args = parser.parse_args()

    roots = [args.measured_root, args.rnet_root, args.rnafold_root, args.assisted_root]
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
    if args.samples < 1:
        raise ValueError("--samples must be positive")

    measured_ids = ids_from_list(args.measured_id_list)
    rnet_ids = ids_in(args.rnet_root, ".shape.tsv")
    rnafold_ids = ids_in(args.rnafold_root, ".bpseq")
    assisted_ids = ids_in(args.assisted_root, ".bpseq")
    common = sorted(measured_ids & rnet_ids & rnafold_ids & assisted_ids)
    if len(common) < args.samples:
        raise RuntimeError(f"only {len(common)} matched IDs, need {args.samples}")
    common = common[: args.samples]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rnet_core = args.out_dir / "rnet_core"
    rnet_core.mkdir(exist_ok=True)
    measured_lengths: list[int] = []
    rnet_lengths: list[int] = []
    for ident in common:
        measured_seq, measured_rows = read_shape(args.measured_root / f"{ident}.shape.tsv")
        rnet_seq, rnet_rows = read_shape(args.rnet_root / f"{ident}.shape.tsv")
        stop = len(rnet_rows) - args.trim_3
        if stop <= args.trim_5:
            raise RuntimeError(f"invalid RNet length for {ident}: {len(rnet_rows)}")
        core_seq = rnet_seq[args.trim_5 : stop]
        if core_seq != measured_seq:
            raise RuntimeError(
                f"measured/RNet core mismatch for {ident}: "
                f"measured={len(measured_seq)}, rnet={len(rnet_seq)}"
            )
        core_rows = rnet_rows[args.trim_5 : stop]
        measured_lengths.append(len(measured_rows))
        rnet_lengths.append(len(core_rows))
        out = rnet_core / f"{ident}.shape.tsv"
        out.write_text("".join(f"{i}\t{base}\t{value}\n" for i, (base, value) in enumerate(core_rows, 1)))

    ids_file = args.out_dir / "common_ids.lst"
    ids_file.write_text("".join(f"{ident}\n" for ident in common))
    manifest = {
        "comparison": "measured SN0 2A3_MaP versus RibonanzaNet prediction",
        "measured_root": str(args.measured_root),
        "rnet_root": str(args.rnet_root),
        "rnet_core_root": str(rnet_core),
        "rnafold_root": str(args.rnafold_root),
        "assisted_root": str(args.assisted_root),
        "measured_id_list": str(args.measured_id_list),
        "trimmed_constant_regions": {"five_prime": args.trim_5, "three_prime": args.trim_3},
        "measured_source_count": len(measured_ids),
        "rnet_source_count": len(rnet_ids),
        "rnafold_source_count": len(rnafold_ids),
        "assisted_source_count": len(assisted_ids),
        "matched_intersection_count": len(measured_ids & rnet_ids & rnafold_ids & assisted_ids),
        "selected_count": len(common),
        "length_min": min(measured_lengths),
        "length_max": max(measured_lengths),
        "length_mean": sum(measured_lengths) / len(measured_lengths),
        "common_ids_sha256": sha256(ids_file),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
