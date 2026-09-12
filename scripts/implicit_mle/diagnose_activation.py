#!/usr/bin/env python3
"""Measure whether the implicit-MLE pseudo-energy changes the MAP structure.

This is deliberately a read-only diagnostic: it does not train or modify a
checkpoint.  For every SHAPE sequence it measures the aligned implementation
under alternative rank/NLL and gradient-scaling settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F

from mxfold2 import interface
from mxfold2.fold.shape_layers import Wu
from mxfold2.train import Train


PROFILES = (
    ("rank_aligned", "rank_clipped", "aligned", "none"),
    ("rank_raw_aligned", "rank_raw", "aligned", "none"),
    ("rank_raw_rms", "rank_raw", "aligned", "rms"),
    ("nll_aligned", "nll_wu", "aligned", "none"),
    ("nll_rms", "nll_wu", "aligned", "rms"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-list", type=Path, required=True)
    parser.add_argument("--shape-root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nu-values", default="0.1,1,10,100")
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.0)
    return parser.parse_args()


def resolve_shape_paths(list_path: Path, shape_root: Path | None) -> list[Path]:
    paths: list[Path] = []
    with list_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            candidate = Path(fields[0])
            if not candidate.exists() and shape_root is not None:
                name = candidate.name
                if not name.endswith(".shape.tsv"):
                    name += ".shape.tsv"
                candidate = shape_root / name
            if not candidate.is_file():
                raise FileNotFoundError(
                    f"{list_path}:{line_number}: cannot resolve {fields[0]!r}"
                )
            paths.append(candidate.resolve())
    if not paths:
        raise ValueError(f"No SHAPE files were listed in {list_path}")
    return paths


def read_shape(path: Path) -> tuple[str, torch.Tensor, dict[str, float]]:
    sequence: list[str] = []
    values = [-999.0]  # Match ShapeDataset and traceback's 1-based pair vector.
    expected_position = 1
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(f"{path}:{line_number}: expected 3 columns")
            position, base, value = fields
            if int(position) != expected_position:
                raise ValueError(
                    f"{path}:{line_number}: position {position}, expected {expected_position}"
                )
            expected_position += 1
            sequence.append(base)
            try:
                values.append(float(value))
            except ValueError:
                values.append(-999.0)

    target = torch.tensor(values, dtype=torch.float32)
    real = target[1:]
    valid = real > -10
    n_valid = int(valid.sum())
    stats = {
        "valid_fraction": n_valid / max(len(real), 1),
        "clip_low_fraction": float(((real < 0.01) & valid).sum()) / max(n_valid, 1),
        "clip_high_fraction": float(((real > 1.0) & valid).sum()) / max(n_valid, 1),
    }
    return "".join(sequence), target, stats


def build_model(config: Path, checkpoint: Path, device: torch.device):
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    subparsers = parser.add_subparsers()
    Train.add_args(subparsers)
    model_args = parser.parse_args(
        ["train", f"@{config.resolve()}", "/dev/null"]
    )
    model, _ = Train().build_model(model_args)
    state: Any = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = tuple(k for k in missing if "shape_head" in k)
    other_missing = sorted(set(missing) - set(allowed_missing))
    if other_missing or unexpected:
        raise RuntimeError(
            f"Incompatible checkpoint: missing={other_missing}, unexpected={unexpected}"
        )
    model.to(device)
    model.eval()
    return model, list(missing), list(unexpected)


def paired_tensor(pair: list[int], device: torch.device, grad: bool) -> torch.Tensor:
    return torch.tensor(
        [1.0 if partner > 0 else 0.0 for partner in pair],
        dtype=torch.float32,
        device=device,
        requires_grad=grad,
    )


def rank_objective(
    target: torch.Tensor,
    paired: torch.Tensor,
    *,
    clip: bool,
    tau: float,
    margin: float,
) -> torch.Tensor:
    valid = target > -10
    values = target.clamp(0.01, 1.0) if clip else target
    t = values[valid].to(paired.device)
    p = paired[valid]
    if t.numel() <= 1:
        return p.sum() * 0.0
    weights = p[:, None] * (1.0 - p[None, :])
    weights = weights * (1.0 - torch.eye(t.numel(), device=t.device))
    denominator = weights.sum()
    if denominator <= 0:
        return p.sum() * 0.0
    difference = t[None, :] - t[:, None]
    penalty = F.softplus(-(difference - margin) / tau)
    return (weights * penalty).sum() / (denominator + 1e-8)


def objective_and_gradient(
    objective: str,
    sequence: str,
    target: torch.Tensor,
    pair: list[int],
    device: torch.device,
    wu: Wu,
    tau: float,
    margin: float,
) -> tuple[float, torch.Tensor]:
    paired = paired_tensor(pair, device, grad=True)
    target_device = target.to(device)
    if objective == "rank_clipped":
        loss = rank_objective(
            target_device, paired, clip=True, tau=tau, margin=margin
        )
    elif objective == "rank_raw":
        loss = rank_objective(
            target_device, paired, clip=False, tau=tau, margin=margin
        )
    elif objective == "nll_wu":
        loss = wu([sequence], [paired], [target_device]).sum()
    else:
        raise ValueError(objective)
    gradient = torch.autograd.grad(loss, paired)[0].detach()
    return float(loss.detach()), gradient


def objective_value(
    objective: str,
    sequence: str,
    target: torch.Tensor,
    pair: list[int],
    device: torch.device,
    wu: Wu,
    tau: float,
    margin: float,
) -> float:
    with torch.no_grad():
        paired = paired_tensor(pair, device, grad=False)
        target_device = target.to(device)
        if objective == "rank_clipped":
            value = rank_objective(
                target_device, paired, clip=True, tau=tau, margin=margin
            )
        elif objective == "rank_raw":
            value = rank_objective(
                target_device, paired, clip=False, tau=tau, margin=margin
            )
        else:
            value = wu([sequence], [paired], [target_device]).sum()
    return float(value)


def iter_named_tensors(tree: dict[str, Any], prefix: str = "") -> Iterable[tuple[str, torch.Tensor]]:
    for key in sorted(tree):
        value = tree[key]
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from iter_named_tensors(value, name)
        elif key.startswith("count_") and isinstance(value, torch.Tensor):
            yield name, value


def clone_counts(param: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in iter_named_tensors(param)}


def count_difference(
    predicted: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> tuple[float, float, int]:
    squared = 0.0
    absolute = 0.0
    nonzero = 0
    if predicted.keys() != reference.keys():
        raise RuntimeError("The predicted/reference count tensors have different keys")
    for key in predicted:
        difference = predicted[key] - reference[key]
        squared += float((difference * difference).sum())
        absolute += float(difference.abs().sum())
        nonzero += int((difference != 0).sum())
    return math.sqrt(squared), absolute, nonzero


def normalize_gradient(gradient: torch.Tensor, mode: str) -> tuple[torch.Tensor, float]:
    real = gradient[1:]
    if mode == "none":
        scale = torch.tensor(1.0, device=gradient.device)
    elif mode == "max":
        scale = real.abs().max().clamp_min(1e-12)
    elif mode == "rms":
        scale = real.square().mean().sqrt().clamp_min(1e-12)
    else:
        raise ValueError(mode)
    return gradient / scale, float(scale)


def pair_edges(pair: list[int]) -> set[tuple[int, int]]:
    return {(i, partner) for i, partner in enumerate(pair) if i > 0 and partner > i}


def make_pseudoenergy(
    gradient: torch.Tensor, normalization: str, nu: float
) -> tuple[torch.Tensor, float]:
    normalized, scale = normalize_gradient(gradient, normalization)
    source = normalized[1:]
    return nu * source, scale


def summarize(rows: list[dict[str, Any]], summary_path: Path) -> None:
    groups: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["profile"], row["nu"])].append(row)
    fields = [
        "profile",
        "nu",
        "n",
        "activation_rate",
        "mean_paired_flips",
        "mean_edge_changes",
        "mean_objective_delta",
        "objective_improvement_rate",
        "mean_count_diff_l2",
        "mean_count_diff_l2_div_nu",
        "mean_pseudo_rms",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (profile, nu), values in sorted(groups.items()):
            n = len(values)
            record = {
                "profile": profile,
                "nu": nu,
                "n": n,
                "activation_rate": sum(v["changed"] for v in values) / n,
                "mean_paired_flips": sum(v["paired_flips"] for v in values) / n,
                "mean_edge_changes": sum(v["edge_changes"] for v in values) / n,
                "mean_objective_delta": sum(v["objective_delta"] for v in values) / n,
                "objective_improvement_rate": sum(
                    v["objective_delta"] < 0 for v in values
                ) / n,
                "mean_count_diff_l2": sum(v["count_diff_l2"] for v in values) / n,
                "mean_count_diff_l2_div_nu": sum(
                    v["count_diff_l2_div_nu"] for v in values
                ) / n,
                "mean_pseudo_rms": sum(v["pseudo_rms"] for v in values) / n,
            }
            writer.writerow(record)
            print(json.dumps(record, sort_keys=True))


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    nu_values = [float(value) for value in args.nu_values.split(",")]
    if not nu_values or any(value <= 0 for value in nu_values):
        raise ValueError("--nu-values must contain positive comma-separated values")

    torch.set_num_threads(args.threads)
    interface.set_num_threads(args.threads)
    if args.gpu >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("--gpu was requested but CUDA is unavailable")
        device = torch.device("cuda", args.gpu)
    else:
        device = torch.device("cpu")

    paths = resolve_shape_paths(args.shape_list, args.shape_root)
    rng = random.Random(args.seed)
    if len(paths) > args.max_samples:
        paths = rng.sample(paths, args.max_samples)
    paths.sort()

    model, missing, unexpected = build_model(args.config, args.checkpoint, device)
    wu = Wu().to(device).eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "metadata",
        "config": str(args.config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "shape_list": str(args.shape_list.resolve()),
        "shape_root": None if args.shape_root is None else str(args.shape_root.resolve()),
        "samples": len(paths),
        "seed": args.seed,
        "tau": args.tau,
        "margin": args.margin,
        "nu_values": nu_values,
        "missing_checkpoint_keys": missing,
        "unexpected_checkpoint_keys": unexpected,
        "note": "all profiles drop the L+1 pairedness dummy index at 0",
    }

    rows: list[dict[str, Any]] = []
    with args.output.open("w") as output:
        output.write(json.dumps(metadata, sort_keys=True) + "\n")
        for sample_index, path in enumerate(paths):
            sequence, target, data_stats = read_shape(path)
            if len(sequence) + 1 != target.numel():
                raise RuntimeError(f"Length mismatch in {path}")
            with torch.no_grad():
                _, _, predicted_pairs, param, _ = model(
                    [sequence], return_param=True, return_count=True
                )
            predicted_pair = predicted_pairs[0]
            if len(predicted_pair) != len(sequence) + 1:
                raise RuntimeError(
                    f"Traceback length {len(predicted_pair)} != L+1 for {path}"
                )
            predicted_counts = clone_counts(param[0])
            predicted_edges = pair_edges(predicted_pair)

            gradients: dict[str, tuple[float, torch.Tensor]] = {}
            for objective in {profile[1] for profile in PROFILES}:
                gradients[objective] = objective_and_gradient(
                    objective,
                    sequence,
                    target,
                    predicted_pair,
                    device,
                    wu,
                    args.tau,
                    args.margin,
                )

            for profile, objective, alignment, normalization in PROFILES:
                loss_before, gradient = gradients[objective]
                for nu in nu_values:
                    pseudoenergy, gradient_scale = make_pseudoenergy(
                        gradient, normalization, nu
                    )
                    with torch.no_grad():
                        _, _, reference_pairs, param, _ = model(
                            [sequence],
                            param=param,
                            return_param=True,
                            return_count=True,
                            pseudoenergy=[pseudoenergy],
                        )
                    reference_pair = reference_pairs[0]
                    reference_counts = clone_counts(param[0])
                    diff_l2, diff_l1, diff_nonzero = count_difference(
                        predicted_counts, reference_counts
                    )
                    loss_after = objective_value(
                        objective,
                        sequence,
                        target,
                        reference_pair,
                        device,
                        wu,
                        args.tau,
                        args.margin,
                    )
                    paired_flips = sum(
                        (left > 0) != (right > 0)
                        for left, right in zip(predicted_pair[1:], reference_pair[1:])
                    )
                    edge_changes = len(
                        predicted_edges.symmetric_difference(pair_edges(reference_pair))
                    )
                    pseudo_real = pseudoenergy[: len(sequence)]
                    row = {
                        "type": "sample",
                        "sample_index": sample_index,
                        "shape_file": str(path),
                        "length": len(sequence),
                        **data_stats,
                        "profile": profile,
                        "objective": objective,
                        "alignment": alignment,
                        "normalization": normalization,
                        "nu": nu,
                        "objective_before": loss_before,
                        "objective_after": loss_after,
                        "objective_delta": loss_after - loss_before,
                        "gradient_real_max_abs": float(gradient[1:].abs().max()),
                        "gradient_real_rms": float(gradient[1:].square().mean().sqrt()),
                        "gradient_scale": gradient_scale,
                        "pseudo_max_abs": float(pseudo_real.abs().max()),
                        "pseudo_rms": float(pseudo_real.square().mean().sqrt()),
                        "paired_flips": paired_flips,
                        "edge_changes": edge_changes,
                        "changed": edge_changes > 0,
                        "count_diff_l2": diff_l2,
                        "count_diff_l1": diff_l1,
                        "count_diff_nonzero": diff_nonzero,
                        # Current ADwrapper omits this 1/nu finite-difference factor.
                        "count_diff_l2_div_nu": diff_l2 / nu,
                    }
                    rows.append(row)
                    output.write(json.dumps(row, sort_keys=True) + "\n")
            print(f"completed {sample_index + 1}/{len(paths)}: {path.name}", flush=True)

    summarize(rows, args.output.with_suffix(".summary.csv"))


if __name__ == "__main__":
    main()
