#!/usr/bin/env python3
"""Evaluate the pretraining Mann-Whitney objective at saved checkpoints."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from argparse import ArgumentParser

from mxfold2 import interface
from mxfold2.dataset import ShapeDataset
from mxfold2.loss.nonparametric_rank import mann_whitney_rank_loss
from mxfold2.train import Train


def parse_model_args(config: Path, gpu: int, empty_bpseq_list: Path, threads: int = 4):
    parser = ArgumentParser(fromfile_prefix_chars="@")
    subparsers = parser.add_subparsers(title="Sub-commands")
    Train.add_args(subparsers)
    return parser.parse_args([
        "train", f"@{config}", "--gpu", str(gpu), "--threads", str(threads),
        "--task", "Folding",
        "--param", "unused-eval-weights.pth", str(empty_bpseq_list),
    ])


def read_model_state(path: Path) -> dict[str, torch.Tensor]:
    saved = torch.load(path, map_location="cpu")
    if isinstance(saved, dict) and "model_state_dict" in saved:
        saved = saved["model_state_dict"]
    if not isinstance(saved, dict):
        raise TypeError(f"Unsupported checkpoint format: {path}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shape-list", type=Path, required=True)
    parser.add_argument("--empty-bpseq-list", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--shape-perturb", type=float, default=0.1)
    parser.add_argument("--shape-loss-weight", type=float, default=0.5)
    parser.add_argument("--testsetb-metrics", type=Path, required=True)
    parser.add_argument("--loss-output", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", action="append", required=True, metavar="EPOCH=PATH",
        help="checkpoint specification; repeat for each epoch",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    torch.set_num_threads(args.threads)
    interface.set_num_threads(args.threads)

    device = torch.device("cuda", args.gpu) if args.gpu >= 0 else torch.device("cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {args.gpu} was requested but is unavailable")

    model_args = parse_model_args(args.config, args.gpu, args.empty_bpseq_list, args.threads)
    trainer = Train()
    model, _ = trainer.build_model(model_args)
    model.to(device)
    model.eval()
    expected = model.state_dict()
    expected_keys = set(expected)

    dataset = ShapeDataset(str(args.shape_list), dataset_id=0)
    if len(dataset) != 3000:
        raise RuntimeError(f"Expected 3000 pretraining SHAPE records, got {len(dataset)}")

    checkpoints: list[tuple[int, Path]] = []
    for spec in args.checkpoint:
        epoch_text, sep, path_text = spec.partition("=")
        if not sep:
            raise ValueError(f"Invalid --checkpoint value: {spec}")
        checkpoints.append((int(epoch_text), Path(path_text)))
    if [epoch for epoch, _ in checkpoints] != [0, *range(10, 101, 10)]:
        raise ValueError("Expected checkpoints for epoch 0 and every 10 epochs through 100")

    args.loss_output.parent.mkdir(parents=True, exist_ok=True)
    with args.loss_output.open("w", newline="") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "epoch", "n_sequences", "n_observed_bases",
            "mann_whitney_loss_raw", "mann_whitney_loss_weighted",
            "perturb", "eval_mode", "seed",
        ])
        output.flush()

        for epoch, checkpoint_path in checkpoints:
            if not checkpoint_path.is_file():
                raise FileNotFoundError(checkpoint_path)
            state = read_model_state(checkpoint_path)
            source_keys = set(state)
            optional_shape_head = {
                "zuker.net.shape_head_1d.weight", "zuker.net.shape_head_1d.bias"
            }
            missing = expected_keys - source_keys
            unexpected = source_keys - expected_keys
            if missing or unexpected - optional_shape_head:
                raise RuntimeError(
                    f"Epoch {epoch} checkpoint mismatch: missing={sorted(missing)}, "
                    f"unexpected={sorted(unexpected - optional_shape_head)}"
                )
            shape_mismatch = [
                key for key in expected_keys & source_keys
                if state[key].shape != expected[key].shape
            ]
            if shape_mismatch:
                raise RuntimeError(f"Epoch {epoch} tensor shape mismatch: {shape_mismatch}")
            model.load_state_dict(
                {key: state[key] for key in expected_keys}, strict=True
            )
            model.eval()

            # Use the same rank objective, sample-level averaging, and SHAPE
            # perturbation as pretraining. Eval mode disables dropout; resetting
            # the RNG makes the one perturb-and-MAP sample comparable by epoch.
            torch.manual_seed(args.seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed)

            total_loss = 0.0
            observed_bases = 0
            with torch.no_grad():
                for index in range(len(dataset)):
                    filename, sequence, values = dataset[index]
                    _, _, paired_vectors, _, _ = model(
                        [sequence], return_param=True, return_count=True,
                        perturb=args.shape_perturb,
                    )
                    paired = torch.tensor(
                        [1.0 if partner > 0 else 0.0 for partner in paired_vectors[0]],
                        dtype=torch.float32, device=device,
                    )
                    target = values["target"].to(device)
                    loss = mann_whitney_rank_loss(target, paired)
                    value = float(loss.item())
                    # Exact 1-AUROC spans [0, 1]; 0.5 is the neutral value.
                    if not math.isfinite(value) or not -1e-6 <= value <= 1.0 + 1e-6:
                        raise RuntimeError(
                            f"Invalid epoch-{epoch} rank loss for {filename}: {value}"
                        )
                    total_loss += value
                    observed_bases += int((torch.isfinite(target) & (target > -10)).sum().item())

            mean_loss = total_loss / len(dataset)
            writer.writerow([
                epoch, len(dataset), observed_bases, f"{mean_loss:.9f}",
                f"{mean_loss * args.shape_loss_weight:.9f}",
                args.shape_perturb, "eval_no_dropout", args.seed,
            ])
            output.flush()
            print(f"epoch={epoch} mean_mw_loss={mean_loss:.6f}", flush=True)

    # Join the added pretraining objective loss to the already-computed
    # TestSetB metrics without retaining per-sequence predictions.
    with args.testsetb_metrics.open(newline="") as handle:
        test_rows = list(csv.DictReader(handle, delimiter="\t"))
    with args.loss_output.open(newline="") as handle:
        loss_rows = {int(row["epoch"]): row for row in csv.DictReader(handle, delimiter="\t")}
    test_rows_by_epoch = {int(row["epoch"]): row for row in test_rows}
    expected_epochs = [0, *range(10, 101, 10)]
    if sorted(test_rows_by_epoch) != expected_epochs or sorted(loss_rows) != expected_epochs:
        raise RuntimeError("TestSetB results and pretraining loss epochs do not match")
    fieldnames = list(test_rows[0]) + [
        "n_pretrain_sequences", "n_pretrain_observed_bases",
        "mann_whitney_loss_raw", "mann_whitney_loss_weighted",
    ]
    args.combined_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.combined_output.with_suffix(args.combined_output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for epoch in expected_epochs:
            merged = dict(test_rows_by_epoch[epoch])
            loss_row = loss_rows[epoch]
            merged.update({
                "n_pretrain_sequences": loss_row["n_sequences"],
                "n_pretrain_observed_bases": loss_row["n_observed_bases"],
                "mann_whitney_loss_raw": loss_row["mann_whitney_loss_raw"],
                "mann_whitney_loss_weighted": loss_row["mann_whitney_loss_weighted"],
            })
            writer.writerow(merged)
    temporary.replace(args.combined_output)


if __name__ == "__main__":
    main()
