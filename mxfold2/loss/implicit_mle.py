from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import torch


Alignment = Literal["aligned"]
Normalization = Literal["none", "rms", "max"]
EstimatorScale = Literal["divide_nu", "none"]
DifferenceMode = Literal["forward", "central"]


@dataclass(frozen=True)
class AdaptiveStepConfig:
    enabled: bool = False
    minimum: float = 1e-3
    maximum: float = 1e3
    flip_fraction_low: float = 0.01
    flip_fraction_high: float = 0.10
    factor: float = 2.0
    max_steps: int = 4

    def validate(self) -> None:
        if self.minimum <= 0 or self.maximum < self.minimum:
            raise ValueError("adaptive nu bounds must satisfy 0 < minimum <= maximum")
        if not 0 <= self.flip_fraction_low <= self.flip_fraction_high <= 1:
            raise ValueError("adaptive flip fractions must satisfy 0 <= low <= high <= 1")
        if self.factor <= 1:
            raise ValueError("adaptive nu factor must be greater than 1")
        if self.max_steps < 1:
            raise ValueError("adaptive max_steps must be at least 1")


@dataclass
class ImplicitMLEResult:
    loss: torch.Tensor
    prediction_score: torch.Tensor
    target_score: torch.Tensor
    prediction_structures: list[str]
    target_structures: list[str]
    selected_nu: list[float]
    flip_fraction: list[float]


def prepare_pseudoenergy(
    gradients: Sequence[torch.Tensor],
    sequence_lengths: Sequence[int],
    *,
    nu: float | Sequence[float],
    alignment: Alignment = "aligned",
    normalization: Normalization = "none",
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Convert pairedness gradients into zero-based, L-long energies."""
    if len(gradients) != len(sequence_lengths):
        raise ValueError("gradients and sequence_lengths must have equal length")
    nus = _expand_positive_values(nu, len(gradients), "nu")

    pseudoenergies: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for gradient, length, step_size in zip(gradients, sequence_lengths, nus):
        if gradient.ndim != 1:
            raise ValueError(f"pairedness gradient must be 1D, got {gradient.shape}")
        if alignment != "aligned":
            raise ValueError(f"unknown alignment: {alignment}")
        if gradient.numel() == length + 1:
            # Pair vectors are 1-based and have a dummy value at index 0.
            source = gradient[1:]
        elif gradient.numel() == length:
            source = gradient
        else:
            raise ValueError(
                f"gradient length {gradient.numel()} is incompatible with sequence length {length}"
            )

        if normalization == "none":
            scale = torch.ones((), dtype=source.dtype, device=source.device)
        elif normalization == "rms":
            scale = source.square().mean().sqrt().clamp_min(1e-12)
        elif normalization == "max":
            scale = source.abs().max().clamp_min(1e-12)
        else:
            raise ValueError(f"unknown normalization: {normalization}")

        pseudoenergies.append(step_size * source / scale)
        scales.append(scale)

    return pseudoenergies, scales


class _DetachedImplicitMLE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        n_pred = len(inputs) // 2
        ctx.n_pred = n_pred
        ctx.save_for_backward(*inputs[n_pred:])
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        artificial = []
        for difference in ctx.saved_tensors:
            if grad_output.dim() == 0:
                artificial.append(difference * grad_output)
            else:
                shape = [grad_output.shape[0]] + [1] * (difference.dim() - 1)
                artificial.append(difference * grad_output.view(*shape))
        return (None,) + tuple(artificial) + (None,) * ctx.n_pred


class _PassthroughImplicitMLE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        n_pred = len(inputs) // 2
        ctx.n_pred = n_pred
        ctx.save_for_backward(*inputs[n_pred:])
        return value

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        artificial = []
        for difference in ctx.saved_tensors:
            if grad_output.dim() == 0:
                artificial.append(difference * grad_output)
            else:
                shape = [grad_output.shape[0]] + [1] * (difference.dim() - 1)
                artificial.append(difference * grad_output.view(*shape))
        return (grad_output,) + tuple(artificial) + (None,) * ctx.n_pred


def apply_count_difference_gradient(
    value: torch.Tensor,
    predicted_parameters: Sequence[torch.Tensor],
    positive_counts: Sequence[torch.Tensor],
    negative_counts: Sequence[torch.Tensor],
    *,
    nu: float | Sequence[float],
    normalization_scales: Sequence[torch.Tensor] | None = None,
    estimator_scale: EstimatorScale = "divide_nu",
    propagate_value_gradient: bool = False,
) -> torch.Tensor:
    """Attach a finite-difference gradient to the score parameter tensors.

    If the downstream gradient was divided by a normalization scale ``s``, the
    actual finite-difference displacement is ``nu / s``. Multiplying the count
    difference by ``s / nu`` restores the scale of the original loss gradient.
    """
    if not (
        len(predicted_parameters) == len(positive_counts) == len(negative_counts)
    ):
        raise ValueError("parameter and count tensor lists must have equal length")

    batch_size = int(value.numel()) if value.ndim <= 1 else int(value.shape[0])
    nus = _expand_positive_values(nu, batch_size, "nu")
    if normalization_scales is None:
        scales = [1.0] * batch_size
    else:
        if len(normalization_scales) != batch_size:
            raise ValueError("normalization scales must have one value per sequence")
        scales = [float(scale.detach()) for scale in normalization_scales]

    if estimator_scale == "divide_nu":
        coefficients = [scale / step for scale, step in zip(scales, nus)]
    elif estimator_scale == "none":
        coefficients = [1.0] * batch_size
    else:
        raise ValueError(f"unknown estimator scale: {estimator_scale}")

    differences = []
    for positive, negative in zip(positive_counts, negative_counts):
        if positive.shape != negative.shape:
            raise ValueError("positive and negative count tensors must have equal shape")
        if batch_size == 1:
            coefficient = torch.tensor(
                coefficients[0], dtype=positive.dtype, device=positive.device
            )
        else:
            if positive.ndim == 0 or positive.shape[0] != batch_size:
                raise ValueError(
                    "batched count tensors must have the sequence batch as first dimension"
                )
            coefficient = torch.tensor(
                coefficients, dtype=positive.dtype, device=positive.device
            ).view(batch_size, *([1] * (positive.ndim - 1)))
        differences.append(((positive - negative) * coefficient).detach().clone())

    function = (
        _PassthroughImplicitMLE if propagate_value_gradient else _DetachedImplicitMLE
    )
    input_value = value if propagate_value_gradient else value.detach().clone()
    return function.apply(input_value, *predicted_parameters, *differences)


def estimate_implicit_mle(
    model: Any,
    sequences: list[str],
    objective: Callable[[list[torch.Tensor]], torch.Tensor],
    *,
    perturb: float,
    nu: float,
    alignment: Alignment,
    normalization: Normalization,
    estimator_scale: EstimatorScale,
    difference_mode: DifferenceMode = "forward",
    num_samples: int = 1,
    adaptive: AdaptiveStepConfig | None = None,
    propagate_value_gradient: bool = False,
    reference_model_weights: dict[str, float] | None = None,
) -> ImplicitMLEResult:
    """Estimate an I-MLE gradient using coupled perturb-and-MAP folds."""
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if difference_mode not in ("forward", "central"):
        raise ValueError(f"unknown difference mode: {difference_mode}")
    adaptive = adaptive or AdaptiveStepConfig()
    adaptive.validate()
    if adaptive.enabled and len(sequences) != 1:
        raise ValueError("adaptive nu currently requires batch_size=1")

    sample_losses: list[torch.Tensor] = []
    prediction_scores: list[torch.Tensor] = []
    target_scores: list[torch.Tensor] = []
    selected_nus: list[float] = []
    flip_fractions: list[float] = []
    first_prediction_structures: list[str] = []
    first_target_structures: list[str] = []

    for sample_index in range(num_samples):
        pred_score, pred_structures, pred_pairs, parameters, _ = model(
            sequences,
            return_param=True,
            return_count=True,
            perturb=perturb,
        )
        pred_parameters, pred_counts = collect_parameters_and_counts(parameters)
        paired = [
            torch.tensor(
                [1.0 if partner > 0 else 0.0 for partner in pairs],
                dtype=torch.float32,
                device=pred_score.device,
                requires_grad=True,
            )
            for pairs in pred_pairs
        ]
        values = objective(paired)
        gradients = torch.autograd.grad(
            values.sum(),
            paired,
            retain_graph=propagate_value_gradient,
            create_graph=False,
        )
        gradients = tuple(gradient.detach().clone() for gradient in gradients)

        candidate_nu = float(nu)
        best = None
        best_fraction = 0.0
        best_positive_score = pred_score
        attempts = adaptive.max_steps if adaptive.enabled else 1
        for _ in range(attempts):
            pseudoenergy, scales = prepare_pseudoenergy(
                gradients,
                [len(sequence) for sequence in sequences],
                nu=candidate_nu,
                alignment=alignment,
                normalization=normalization,
            )
            if difference_mode == "central":
                (
                    positive_score,
                    positive_structures,
                    positive_pairs,
                    parameters,
                    _,
                ) = _reference_fold(
                    model,
                    sequences,
                    parameters,
                    [-0.5 * energy for energy in pseudoenergy],
                    reference_model_weights,
                )
                _, positive_counts = collect_parameters_and_counts(parameters)
                target_score, target_structures, negative_pairs, parameters, _ = _reference_fold(
                    model,
                    sequences,
                    parameters,
                    [0.5 * energy for energy in pseudoenergy],
                    reference_model_weights,
                )
                _, negative_counts = collect_parameters_and_counts(parameters)
                fractions = paired_flip_fraction(
                    positive_pairs, negative_pairs, [len(s) for s in sequences]
                )
            else:
                positive_counts = pred_counts
                positive_pairs = pred_pairs
                positive_score = pred_score
                positive_structures = pred_structures
                target_score, target_structures, negative_pairs, parameters, _ = _reference_fold(
                    model,
                    sequences,
                    parameters,
                    pseudoenergy,
                    reference_model_weights,
                )
                _, negative_counts = collect_parameters_and_counts(parameters)
                fractions = paired_flip_fraction(
                    pred_pairs, negative_pairs, [len(s) for s in sequences]
                )

            mean_fraction = sum(fractions) / len(fractions)
            distance = _distance_to_interval(
                mean_fraction,
                adaptive.flip_fraction_low,
                adaptive.flip_fraction_high,
            )
            if best is None or distance < best[0]:
                best = (
                    distance,
                    candidate_nu,
                    scales,
                    positive_counts,
                    negative_counts,
                    positive_structures,
                    target_score,
                    target_structures,
                )
                best_fraction = mean_fraction
                best_positive_score = positive_score
            if not adaptive.enabled or distance == 0:
                break
            if mean_fraction < adaptive.flip_fraction_low:
                next_nu = min(candidate_nu * adaptive.factor, adaptive.maximum)
            else:
                next_nu = max(candidate_nu / adaptive.factor, adaptive.minimum)
            if next_nu == candidate_nu:
                break
            candidate_nu = next_nu

        assert best is not None
        (
            _,
            chosen_nu,
            scales,
            positive_counts,
            negative_counts,
            prediction_structures,
            target_score,
            target_structures,
        ) = best
        sample_loss = apply_count_difference_gradient(
            values,
            pred_parameters,
            positive_counts,
            negative_counts,
            nu=chosen_nu,
            normalization_scales=scales,
            estimator_scale=estimator_scale,
            propagate_value_gradient=propagate_value_gradient,
        )
        sample_losses.append(sample_loss)
        prediction_scores.append(best_positive_score)
        target_scores.append(target_score)
        selected_nus.append(chosen_nu)
        flip_fractions.append(best_fraction)
        if sample_index == 0:
            first_prediction_structures = prediction_structures
            first_target_structures = target_structures

    logging.debug(
        "I-MLE mode=%s samples=%d nu=%s paired_flip_fraction=%s",
        difference_mode,
        num_samples,
        selected_nus,
        flip_fractions,
    )
    return ImplicitMLEResult(
        loss=torch.stack(sample_losses).mean(dim=0),
        prediction_score=torch.stack(prediction_scores).mean(dim=0),
        target_score=torch.stack(target_scores).mean(dim=0),
        prediction_structures=first_prediction_structures,
        target_structures=first_target_structures,
        selected_nu=selected_nus,
        flip_fraction=flip_fractions,
    )


def _reference_fold(
    model: Any,
    sequences: list[str],
    parameters: list[dict[str, Any]],
    pseudoenergy: list[torch.Tensor],
    reference_model_weights: dict[str, float] | None,
):
    if not reference_model_weights:
        return model(
            sequences,
            param=parameters,
            return_param=True,
            return_count=True,
            pseudoenergy=pseudoenergy,
        )

    missing = [name for name in reference_model_weights if not hasattr(model, name)]
    if missing:
        raise ValueError(f"reference model weights are unavailable: {missing}")

    original = {name: getattr(model, name) for name in reference_model_weights}
    try:
        for name, value in reference_model_weights.items():
            setattr(model, name, value)
        _, structures, _ = model(
            sequences,
            param=parameters,
            pseudoenergy=pseudoenergy,
        )
    finally:
        for name, value in original.items():
            setattr(model, name, value)

    return model(
        sequences,
        param=parameters,
        return_param=True,
        return_count=True,
        constraint=structures,
    )


def collect_parameters_and_counts(
    parameters: list[dict[str, Any]],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    score_parameters: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    for key in sorted(parameters[0]):
        value = parameters[0][key]
        if key.startswith("score_"):
            score_parameters.append(torch.vstack([item[key] for item in parameters]))
        elif key.startswith("count_"):
            counts.append(torch.vstack([item[key] for item in parameters]))
        elif isinstance(value, dict):
            for nested_key in sorted(value):
                if nested_key.startswith("score_"):
                    score_parameters.append(
                        torch.vstack([item[key][nested_key] for item in parameters])
                    )
                elif nested_key.startswith("count_"):
                    counts.append(
                        torch.vstack([item[key][nested_key] for item in parameters])
                    )
    return score_parameters, counts


def paired_flip_fraction(
    first_pairs: Sequence[Sequence[int]],
    second_pairs: Sequence[Sequence[int]],
    sequence_lengths: Sequence[int],
) -> list[float]:
    if not (len(first_pairs) == len(second_pairs) == len(sequence_lengths)):
        raise ValueError("pair lists and sequence_lengths must have equal length")
    fractions = []
    for first, second, length in zip(first_pairs, second_pairs, sequence_lengths):
        first_aligned = _align_pairs(first, length)
        second_aligned = _align_pairs(second, length)
        changes = sum((a > 0) != (b > 0) for a, b in zip(first_aligned, second_aligned))
        fractions.append(changes / length if length else 0.0)
    return fractions


def _align_pairs(pairs: Sequence[int], length: int) -> Sequence[int]:
    if len(pairs) == length + 1:
        return pairs[1:]
    if len(pairs) == length:
        return pairs
    raise ValueError(f"pair vector length {len(pairs)} does not match sequence length {length}")


def _expand_positive_values(
    values: float | Sequence[float], length: int, name: str
) -> list[float]:
    if isinstance(values, (float, int)):
        expanded = [float(values)] * length
    else:
        expanded = [float(value) for value in values]
        if len(expanded) != length:
            raise ValueError(f"{name} must have one value per sequence")
    if any(value <= 0 for value in expanded):
        raise ValueError(f"{name} values must be positive")
    return expanded


def _distance_to_interval(value: float, low: float, high: float) -> float:
    if value < low:
        return low - value
    if value > high:
        return value - high
    return 0.0
