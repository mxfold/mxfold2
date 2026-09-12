#!/usr/bin/env python3
"""Reproducible toy correctness checks for SHAPE rank loss and I-MLE.

The toy fold enumerates every canonical, pseudoknot-free structure of a short
RNA.  Its trainable parameters are one score per enumerated structure.  This
keeps the MAP oracle discrete while making the exact Gibbs expectation small
enough to calculate directly.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from mxfold2.loss.implicit_mle import estimate_implicit_mle
from mxfold2.loss.shape_rank_loss import ShapeRankLoss


CANONICAL_PAIRS = {"AU", "UA", "GC", "CG", "GU", "UG"}
DEFAULT_SEQUENCE = "GGGAAACCC"
DEFAULT_TARGET_PAIRS = ((0, 8), (1, 7), (2, 6))
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def enumerate_structures(
    sequence: str, min_hairpin_unpaired: int = 2
) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate canonical non-crossing structures, including all-unpaired."""

    @lru_cache(maxsize=None)
    def interval(left: int, right: int) -> tuple[tuple[tuple[int, int], ...], ...]:
        if left > right:
            return ((),)

        result = list(interval(left + 1, right))
        first_partner = left + min_hairpin_unpaired + 1
        for partner in range(first_partner, right + 1):
            if sequence[left] + sequence[partner] not in CANONICAL_PAIRS:
                continue
            for inside in interval(left + 1, partner - 1):
                for after in interval(partner + 1, right):
                    result.append(((left, partner),) + inside + after)
        return tuple(result)

    return list(interval(0, len(sequence) - 1))


def paired_features(
    structures: Sequence[Sequence[tuple[int, int]]], length: int
) -> torch.Tensor:
    features = torch.zeros((len(structures), length), dtype=torch.float32)
    for row, structure in enumerate(structures):
        for left, right in structure:
            features[row, left] = 1.0
            features[row, right] = 1.0
    return features


def pair_vector(structure: Sequence[tuple[int, int]], length: int) -> list[int]:
    result = [0] * length
    for left, right in structure:
        result[left] = right + 1
        result[right] = left + 1
    return result


def dot_bracket(structure: Sequence[tuple[int, int]], length: int) -> str:
    result = ["."] * length
    for left, right in structure:
        result[left] = "("
        result[right] = ")"
    return "".join(result)


class EnumeratedFold(torch.nn.Module):
    """Categorical perturb-and-MAP oracle compatible with estimate_implicit_mle."""

    def __init__(
        self,
        sequence: str = DEFAULT_SEQUENCE,
        initial_scores: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.sequence = sequence
        self.structures = enumerate_structures(sequence)
        self.features = paired_features(self.structures, len(sequence))
        if initial_scores is None:
            initial_scores = torch.zeros(len(self.structures), dtype=torch.float32)
        if initial_scores.shape != (len(self.structures),):
            raise ValueError("initial_scores must have one value per structure")
        self.structure_scores = torch.nn.Parameter(initial_scores.detach().clone())

    def _gumbel(self, perturb: float) -> torch.Tensor:
        if perturb == 0.0:
            return torch.zeros_like(self.structure_scores)
        uniform = torch.rand_like(self.structure_scores).clamp_(1e-7, 1.0 - 1e-7)
        return -torch.log(-torch.log(uniform)) * perturb

    def forward(
        self,
        sequences: list[str],
        return_param: bool = False,
        return_count: bool = False,
        param=None,
        pseudoenergy: list[torch.Tensor] | None = None,
        perturb: float = 0.0,
        **kwargs,
    ):
        del kwargs
        if any(sequence != self.sequence for sequence in sequences):
            raise ValueError("EnumeratedFold only accepts its configured sequence")
        if param is None:
            param = [
                {"score_structure": self.structure_scores + self._gumbel(perturb)}
                for _ in sequences
            ]

        scores = []
        strings = []
        pair_vectors = []
        for batch_index in range(len(sequences)):
            raw_scores = param[batch_index]["score_structure"]
            map_scores = raw_scores
            if pseudoenergy is not None:
                energy = pseudoenergy[batch_index].to(raw_scores.device)
                map_scores = map_scores - self.features.to(raw_scores.device) @ energy
            selected = int(map_scores.detach().argmax())
            count = F.one_hot(
                torch.tensor(selected, device=raw_scores.device),
                num_classes=len(self.structures),
            ).to(raw_scores.dtype)
            if return_count:
                param[batch_index]["count_structure"] = count
            scores.append((raw_scores * count).sum())
            structure = self.structures[selected]
            strings.append(dot_bracket(structure, len(self.sequence)))
            pair_vectors.append(pair_vector(structure, len(self.sequence)))

        score_tensor = torch.stack(scores)
        if return_param:
            return score_tensor, strings, pair_vectors, param, param
        return score_tensor, strings, pair_vectors


def target_feature(length: int = len(DEFAULT_SEQUENCE)) -> torch.Tensor:
    return paired_features([DEFAULT_TARGET_PAIRS], length)[0]


def linear_hamming_objective(target: torch.Tensor):
    # Hamming(paired, target) = constant + dot(1 - 2*target, paired).
    coefficient = 1.0 - 2.0 * target

    def objective(paired: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(
            [
                target.sum().to(pair.device)
                + (coefficient.to(pair.device) * pair).sum()
                for pair in paired
            ]
        )

    return objective


def exact_expected_gradient(
    scores: torch.Tensor, costs: torch.Tensor, temperature: float
) -> tuple[float, torch.Tensor]:
    differentiable_scores = scores.detach().clone().requires_grad_(True)
    probabilities = torch.softmax(differentiable_scores / temperature, dim=0)
    expectation = (probabilities * costs).sum()
    gradient = torch.autograd.grad(expectation, differentiable_scores)[0]
    return float(expectation), gradient.detach()


def numerical_expected_gradient(
    scores: torch.Tensor, costs: torch.Tensor, temperature: float, epsilon: float = 1e-3
) -> torch.Tensor:
    values = []
    for index in range(scores.numel()):
        offset = torch.zeros_like(scores)
        offset[index] = epsilon
        plus = (torch.softmax((scores + offset) / temperature, dim=0) * costs).sum()
        minus = (torch.softmax((scores - offset) / temperature, dim=0) * costs).sum()
        values.append((plus - minus) / (2.0 * epsilon))
    return torch.stack(values)


def cosine_similarity(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(F.cosine_similarity(first.flatten(), second.flatten(), dim=0))


def relative_error(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    return float((estimate - reference).norm() / reference.norm().clamp_min(1e-12))


def initial_toy_scores(model: EnumeratedFold) -> torch.Tensor:
    index = torch.arange(len(model.structures), dtype=torch.float32)
    pair_count = model.features.sum(dim=1) / 2.0
    return 0.15 * torch.cos(1.7 * index) - 0.04 * pair_count


def gradient_sanity_check(
    seeds: Sequence[int], samples: int, temperature: float = 0.7, nu: float = 0.1
) -> dict:
    template = EnumeratedFold()
    scores = initial_toy_scores(template)
    target = target_feature()
    costs = (template.features - target).abs().sum(dim=1)
    expectation, exact = exact_expected_gradient(scores, costs, temperature)
    numerical = numerical_expected_gradient(scores, costs, temperature)

    per_seed = []
    estimates = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = EnumeratedFold(initial_scores=scores)
        estimate = estimate_implicit_mle(
            model,
            [DEFAULT_SEQUENCE],
            linear_hamming_objective(target),
            perturb=temperature,
            nu=nu,
            alignment="aligned",
            normalization="none",
            estimator_scale="divide_nu",
            difference_mode="central",
            num_samples=samples,
        )
        estimate.loss.sum().backward()
        gradient = model.structure_scores.grad.detach().clone()
        estimates.append(gradient)
        per_seed.append(
            {
                "seed": seed,
                "cosine": cosine_similarity(gradient, exact),
                "relative_error": relative_error(gradient, exact),
            }
        )

    mean_estimate = torch.stack(estimates).mean(dim=0)
    return {
        "sequence": DEFAULT_SEQUENCE,
        "num_structures": len(template.structures),
        "temperature": temperature,
        "nu": nu,
        "samples_per_seed": samples,
        "exact_expectation": expectation,
        "numerical_vs_exact_cosine": cosine_similarity(numerical, exact),
        "numerical_vs_exact_relative_error": relative_error(numerical, exact),
        "imle_per_seed": per_seed,
        "imle_mean_cosine": cosine_similarity(mean_estimate, exact),
        "imle_mean_relative_error": relative_error(mean_estimate, exact),
    }


def rank_gradient_sanity_check(
    seeds: Sequence[int], samples: int, temperature: float = 0.7, nu: float = 0.1
) -> dict:
    """Compare I-MLE with the exact Gibbs gradient of the new rank objective."""
    template = EnumeratedFold()
    scores = initial_toy_scores(template)
    target = target_feature()
    shape = torch.where(target > 0, 0.05, 1.0)
    template_rank = ShapeRankLoss(template, tau=0.2)
    costs = torch.tensor(
        [
            float(template_rank._per_sequence_rank_loss(shape, feature))
            for feature in template.features
        ]
    )
    expectation, exact = exact_expected_gradient(scores, costs, temperature)
    numerical = numerical_expected_gradient(scores, costs, temperature)

    per_seed = []
    estimates = []
    for seed in seeds:
        torch.manual_seed(seed)
        model = EnumeratedFold(initial_scores=scores)
        rank = ShapeRankLoss(model, tau=0.2)

        def objective(paired: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(
                [rank._per_sequence_rank_loss(shape, pair) for pair in paired]
            )

        estimate = estimate_implicit_mle(
            model,
            [DEFAULT_SEQUENCE],
            objective,
            perturb=temperature,
            nu=nu,
            alignment="aligned",
            normalization="none",
            estimator_scale="divide_nu",
            difference_mode="central",
            num_samples=samples,
        )
        estimate.loss.sum().backward()
        gradient = model.structure_scores.grad.detach().clone()
        estimates.append(gradient)
        per_seed.append(
            {
                "seed": seed,
                "cosine": cosine_similarity(gradient, exact),
                "relative_error": relative_error(gradient, exact),
            }
        )

    mean_estimate = torch.stack(estimates).mean(dim=0)
    return {
        "objective": "pairwise SHAPE-order rank loss",
        "exact_expectation": expectation,
        "samples_per_seed": samples,
        "numerical_vs_exact_cosine": cosine_similarity(numerical, exact),
        "numerical_vs_exact_relative_error": relative_error(numerical, exact),
        "imle_per_seed": per_seed,
        "imle_mean_cosine": cosine_similarity(mean_estimate, exact),
        "imle_mean_relative_error": relative_error(mean_estimate, exact),
    }


def rank_loss_check() -> dict:
    model = EnumeratedFold()
    rank = ShapeRankLoss(model, tau=0.2)
    target = target_feature()
    shape = torch.where(target > 0, 0.05, 1.0)
    losses = torch.tensor(
        [
            float(rank._per_sequence_rank_loss(shape, feature))
            for feature in model.features
        ]
    )
    target_index = model.structures.index(DEFAULT_TARGET_PAIRS)
    nondegenerate_wrong = [
        index
        for index, feature in enumerate(model.features)
        if index != target_index and 0 < float(feature.sum()) < len(DEFAULT_SEQUENCE)
    ]
    wrong_losses = losses[nondegenerate_wrong]
    all_wrong_losses = torch.cat((losses[:target_index], losses[target_index + 1 :]))
    all_unpaired_index = model.structures.index(())
    correct = float(losses[target_index])
    all_unpaired = float(losses[all_unpaired_index])
    return {
        "correct": correct,
        "best_nondegenerate_wrong": float(wrong_losses.min()),
        "mean_nondegenerate_wrong": float(wrong_losses.mean()),
        "best_wrong_including_degenerate": float(all_wrong_losses.min()),
        "all_unpaired": all_unpaired,
        "correct_below_every_nondegenerate_wrong": bool(correct < float(wrong_losses.min())),
        "correct_below_every_wrong": bool(correct < float(all_wrong_losses.min())),
        "correct_below_all_unpaired": bool(correct < all_unpaired),
    }


def target_score_gap(model: EnumeratedFold) -> tuple[float, int]:
    target_index = model.structures.index(DEFAULT_TARGET_PAIRS)
    scores = model.structure_scores.detach()
    wrong = torch.cat((scores[:target_index], scores[target_index + 1 :]))
    return float(scores[target_index] - wrong.max()), target_index


def one_step_check(
    seeds: Sequence[int], samples: int, learning_rate: float = 0.5
) -> list[dict]:
    results = []
    target = target_feature()
    shape = torch.where(target > 0, 0.05, 1.0)
    for seed in seeds:
        torch.manual_seed(seed)
        model = EnumeratedFold(initial_scores=initial_toy_scores(EnumeratedFold()))
        rank = ShapeRankLoss(model, tau=0.2)

        def objective(paired: list[torch.Tensor]) -> torch.Tensor:
            return torch.stack(
                [rank._per_sequence_rank_loss(shape, pair) for pair in paired]
            )

        before_gap, target_index = target_score_gap(model)
        before_score = float(model.structure_scores[target_index])
        estimate = estimate_implicit_mle(
            model,
            [DEFAULT_SEQUENCE],
            objective,
            perturb=0.7,
            nu=0.1,
            alignment="aligned",
            normalization="none",
            estimator_scale="divide_nu",
            difference_mode="central",
            num_samples=samples,
        )
        estimate.loss.sum().backward()
        with torch.no_grad():
            model.structure_scores -= learning_rate * model.structure_scores.grad
        after_gap, _ = target_score_gap(model)
        after_score = float(model.structure_scores[target_index])
        results.append(
            {
                "seed": seed,
                "objective": "pairwise SHAPE-order rank loss",
                "target_score_delta": after_score - before_score,
                "target_vs_best_wrong_gap_delta": after_gap - before_gap,
                "pass": after_score > before_score and after_gap > before_gap,
            }
        )
    return results


def base_pair_f1(
    predicted: Iterable[tuple[int, int]], reference: Iterable[tuple[int, int]]
) -> float:
    predicted_set = set(predicted)
    reference_set = set(reference)
    if not predicted_set and not reference_set:
        return 1.0
    true_positive = len(predicted_set & reference_set)
    precision = true_positive / len(predicted_set) if predicted_set else 0.0
    recall = true_positive / len(reference_set) if reference_set else 0.0
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def sample_wu_shape(seed: int, target: torch.Tensor) -> torch.Tensor:
    """Sample the GEV/Gamma distributions used by mxfold2.fold.shape_layers.Wu."""
    generator = torch.Generator().manual_seed(seed)
    uniform = torch.rand(target.shape, generator=generator).clamp_(1e-7, 1.0 - 1e-7)
    xi, mu, sigma = 0.774, 0.078, 0.083
    paired = mu + sigma / xi * ((-torch.log(uniform)).pow(-xi) - 1.0)
    # torch.distributions does not accept a generator, so isolate its RNG state.
    with torch.random.fork_rng():
        torch.manual_seed(seed + 10_000)
        unpaired = torch.distributions.Gamma(1.006, 1.404).sample(target.shape)
    return torch.where(target > 0, paired, unpaired).clamp(0.01, 1.0)


def train_rank_toy(
    shape: torch.Tensor,
    seed: int,
    steps: int,
    samples: int,
    learning_rate: float = 0.35,
) -> dict:
    torch.manual_seed(seed)
    model = EnumeratedFold()
    loss_module = ShapeRankLoss(
        model,
        perturb=0.7,
        nu=0.2,
        tau=0.2,
        alignment="aligned",
        normalization="none",
        estimator_scale="divide_nu",
        difference_mode="central",
        num_samples=samples,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    initial_gap, _ = target_score_gap(model)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = loss_module([DEFAULT_SEQUENCE], [shape])
        loss.backward()
        optimizer.step()

    final_gap, target_index = target_score_gap(model)
    selected = int(model.structure_scores.detach().argmax())
    f1 = base_pair_f1(model.structures[selected], DEFAULT_TARGET_PAIRS)
    rank_losses = torch.tensor(
        [
            float(loss_module._per_sequence_rank_loss(shape, feature))
            for feature in model.features
        ]
    )
    nondegenerate = torch.tensor(
        [0 < float(feature.sum()) < len(DEFAULT_SEQUENCE) for feature in model.features]
    )
    nondegenerate_indices = torch.where(nondegenerate)[0]
    oracle_index = int(
        nondegenerate_indices[rank_losses[nondegenerate_indices].argmin()]
    )
    global_oracle_index = int(rank_losses.argmin())
    return {
        "seed": seed,
        "f1": f1,
        "exact_structure": selected == target_index,
        "predicted": dot_bracket(model.structures[selected], len(DEFAULT_SEQUENCE)),
        "target_score_gap_delta": final_gap - initial_gap,
        "nondegenerate_oracle_rank_f1": base_pair_f1(
            model.structures[oracle_index], DEFAULT_TARGET_PAIRS
        ),
        "global_rank_min_is_degenerate": not bool(nondegenerate[global_oracle_index]),
        "shape": [round(float(value), 4) for value in shape],
    }


def learning_checks(seeds: Sequence[int], steps: int, samples: int) -> dict:
    target = target_feature()
    perfect_shape = torch.where(target > 0, 0.05, 1.0)
    fake = [train_rank_toy(perfect_shape, seed, steps, samples) for seed in seeds]
    noisy = [
        train_rank_toy(sample_wu_shape(seed, target), seed, steps, samples)
        for seed in seeds
    ]
    return {
        "fake_shape": {
            "paired": 0.05,
            "unpaired": 1.0,
            "per_seed": fake,
            "mean_f1": sum(item["f1"] for item in fake) / len(fake),
            "exact_recovery_rate": sum(item["exact_structure"] for item in fake)
            / len(fake),
        },
        "wu_like_noisy_shape": {
            "distribution": "paired=GEV(0.774,0.078,0.083), unpaired=Gamma(1.006,1.404)",
            "per_seed": noisy,
            "mean_f1": sum(item["f1"] for item in noisy) / len(noisy),
            "exact_recovery_rate": sum(item["exact_structure"] for item in noisy)
            / len(noisy),
            "mean_nondegenerate_oracle_rank_f1": sum(
                item["nondegenerate_oracle_rank_f1"] for item in noisy
            )
            / len(noisy),
        },
    }


def run_checks(
    seeds: Sequence[int] = DEFAULT_SEEDS,
    gradient_samples: int = 1000,
    update_samples: int = 500,
    train_samples: int = 64,
    train_steps: int = 80,
) -> dict:
    gradient = gradient_sanity_check(seeds, gradient_samples)
    rank_gradient = rank_gradient_sanity_check(seeds, gradient_samples)
    rank = rank_loss_check()
    one_step = one_step_check(seeds, update_samples)
    learning = learning_checks(seeds, train_steps, train_samples)
    required = {
        "gradient_direction": gradient["imle_mean_cosine"] >= 0.9,
        "gradient_relative_error": gradient["imle_mean_relative_error"] <= 0.5,
        "rank_gradient_direction": rank_gradient["imle_mean_cosine"] >= 0.9,
        "rank_nondegenerate_ordering": rank[
            "correct_below_every_nondegenerate_wrong"
        ],
        "rank_all_structure_ordering": rank["correct_below_every_wrong"],
        "one_step_all_seeds": all(item["pass"] for item in one_step),
        "fake_shape_mean_f1": learning["fake_shape"]["mean_f1"] >= 0.8,
        "noisy_shape_mean_f1": learning["wu_like_noisy_shape"]["mean_f1"] >= 0.5,
        "five_seeds": len(seeds) >= 5,
    }
    additional = {"degenerate_rank_ordering": rank["correct_below_all_unpaired"]}
    return {
        "gradient_sanity": gradient,
        "rank_gradient_sanity": rank_gradient,
        "rank_loss": rank,
        "one_step_update": one_step,
        **learning,
        "required_checks": required,
        "additional_edge_checks": additional,
        "required_all_pass": all(required.values()),
        "all_including_edge_checks_pass": all(required.values()) and all(additional.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-samples", type=int, default=1000)
    parser.add_argument("--update-samples", type=int, default=500)
    parser.add_argument("--train-samples", type=int, default=64)
    parser.add_argument("--train-steps", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = run_checks(
        args.seeds,
        args.gradient_samples,
        args.update_samples,
        args.train_samples,
        args.train_steps,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    return int(args.strict and not report["all_including_edge_checks_pass"])


if __name__ == "__main__":
    raise SystemExit(main())
