import unittest

import torch

from mxfold2.loss.implicit_mle import (
    AdaptiveStepConfig,
    apply_count_difference_gradient,
    estimate_implicit_mle,
    paired_flip_fraction,
    prepare_pseudoenergy,
)
from mxfold2.loss.shape_rank_loss import ShapeRankLoss


class PreparePseudoenergyTest(unittest.TestCase):
    def test_aligned_drops_dummy_index(self):
        gradient = torch.tensor([99.0, 1.0, 2.0, 3.0])
        pseudo, scale = prepare_pseudoenergy(
            [gradient], [3], nu=2.0, alignment="aligned", normalization="none"
        )
        torch.testing.assert_close(pseudo[0], torch.tensor([2.0, 4.0, 6.0]))
        torch.testing.assert_close(scale[0], torch.tensor(1.0))

    def test_rms_normalization_sets_pseudoenergy_rms_to_nu(self):
        gradient = torch.tensor([0.0, 1.0, 2.0, 3.0])
        pseudo, scale = prepare_pseudoenergy(
            [gradient], [3], nu=0.1, alignment="aligned", normalization="rms"
        )
        self.assertAlmostEqual(float(pseudo[0].square().mean().sqrt()), 0.1, places=6)
        self.assertAlmostEqual(float(scale[0]), (14.0 / 3.0) ** 0.5, places=6)


class CountDifferenceGradientTest(unittest.TestCase):
    def test_count_difference_is_divided_by_nu(self):
        parameter = torch.tensor([[1.0, 2.0]], requires_grad=True)
        predicted = torch.tensor([[5.0, 1.0]])
        reference = torch.tensor([[1.0, 3.0]])
        value = torch.tensor([0.5])
        loss = apply_count_difference_gradient(
            value,
            [parameter],
            [predicted],
            [reference],
            nu=2.0,
            estimator_scale="divide_nu",
        )
        loss.sum().backward()
        torch.testing.assert_close(parameter.grad, torch.tensor([[2.0, -1.0]]))

    def test_rms_scale_restores_effective_finite_difference_step(self):
        parameter = torch.tensor([[1.0, 2.0]], requires_grad=True)
        loss = apply_count_difference_gradient(
            torch.tensor([0.5]),
            [parameter],
            [torch.tensor([[5.0, 1.0]])],
            [torch.tensor([[1.0, 3.0]])],
            nu=2.0,
            normalization_scales=[torch.tensor(0.5)],
        )
        loss.sum().backward()
        torch.testing.assert_close(parameter.grad, torch.tensor([[1.0, -0.5]]))

    def test_sequence_specific_nu_scales_each_batch_row(self):
        parameter = torch.tensor([[1.0], [2.0]], requires_grad=True)
        loss = apply_count_difference_gradient(
            torch.tensor([0.5, 0.5]),
            [parameter],
            [torch.tensor([[2.0], [2.0]])],
            [torch.tensor([[0.0], [0.0]])],
            nu=[1.0, 2.0],
        )
        loss.sum().backward()
        torch.testing.assert_close(parameter.grad, torch.tensor([[2.0], [1.0]]))

    def test_legacy_estimator_scale_is_available_for_ablation(self):
        parameter = torch.tensor([[1.0, 2.0]], requires_grad=True)
        predicted = torch.tensor([[5.0, 1.0]])
        reference = torch.tensor([[1.0, 3.0]])
        loss = apply_count_difference_gradient(
            torch.tensor([0.5]),
            [parameter],
            [predicted],
            [reference],
            nu=2.0,
            estimator_scale="none",
        )
        loss.sum().backward()
        torch.testing.assert_close(parameter.grad, torch.tensor([[4.0, -2.0]]))

    def test_value_gradient_can_pass_through_for_trainable_shape_model(self):
        parameter = torch.tensor([[1.0]], requires_grad=True)
        value = torch.tensor([0.5], requires_grad=True)
        loss = apply_count_difference_gradient(
            value,
            [parameter],
            [torch.tensor([[3.0]])],
            [torch.tensor([[1.0]])],
            nu=2.0,
            propagate_value_gradient=True,
        )
        loss.sum().backward()
        torch.testing.assert_close(parameter.grad, torch.tensor([[1.0]]))
        torch.testing.assert_close(value.grad, torch.tensor([1.0]))


class RankTargetTest(unittest.TestCase):
    def test_rank_is_invariant_to_monotone_target_rescaling(self):
        target = torch.tensor([-999.0, 0.0, 2.0])
        paired = torch.tensor([0.0, 1.0, 0.0])
        raw = ShapeRankLoss(model=torch.nn.Linear(1, 1), clip_targets=False)
        clipped = ShapeRankLoss(model=torch.nn.Linear(1, 1), clip_targets=True)
        raw_loss = raw._per_sequence_rank_loss(target, paired)
        clipped_loss = clipped._per_sequence_rank_loss(target, paired)
        torch.testing.assert_close(raw_loss, clipped_loss)

    def test_rank_delta_ignores_ambiguous_shape_pairs(self):
        target = torch.tensor([0.10, 0.12])
        paired = torch.tensor([1.0, 0.0], requires_grad=True)
        rank = ShapeRankLoss(model=torch.nn.Linear(1, 1), rank_delta=0.05)
        loss = rank._per_sequence_rank_loss(target, paired)
        loss.backward()
        torch.testing.assert_close(loss, torch.tensor(0.0))
        torch.testing.assert_close(paired.grad, torch.zeros_like(paired))

    def test_all_unpaired_has_gradient_toward_shape_ordering(self):
        target = torch.tensor([0.05, 1.0])
        paired = torch.tensor([0.0, 0.0], requires_grad=True)
        rank = ShapeRankLoss(model=torch.nn.Linear(1, 1), tau=0.2)
        loss = rank._per_sequence_rank_loss(target, paired)
        loss.backward()
        self.assertAlmostEqual(float(loss), torch.log(torch.tensor(2.0)).item(), places=6)
        self.assertLess(float(paired.grad[0]), 0.0)
        self.assertGreater(float(paired.grad[1]), 0.0)


class _FakeFold(torch.nn.Module):
    """Small MAP oracle for exercising the estimator without the C++ backend."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.05]))

    def forward(
        self,
        seq,
        return_param=False,
        return_count=False,
        param=None,
        pseudoenergy=None,
        perturb=0.0,
        **kwargs,
    ):
        del kwargs
        if param is None:
            noise = torch.tensor([perturb], device=self.weight.device)
            param = [{"score_pair": self.weight + noise} for _ in seq]
        scores = []
        structures = []
        pairs = []
        for index, sequence in enumerate(seq):
            penalty = 0.0 if pseudoenergy is None else float(pseudoenergy[index].mean())
            is_paired = float(param[index]["score_pair"].detach()) - penalty > 0
            pair = [0] + ([2, 1] if is_paired else [0, 0])
            count = torch.tensor([1.0 if is_paired else 0.0], device=self.weight.device)
            param[index]["count_pair"] = count
            scores.append(param[index]["score_pair"].sum() * count)
            structures.append("()" if is_paired else "..")
            pairs.append(pair)
        score = torch.stack(scores)
        if return_param:
            return score, structures, pairs, param, param
        return score, structures, pairs


class EstimatorTest(unittest.TestCase):
    @staticmethod
    def objective(paired):
        return torch.stack([pair[1:].mean() for pair in paired])

    def test_paired_flip_fraction_drops_dummy(self):
        value = paired_flip_fraction([[0, 2, 1]], [[0, 0, 0]], [2])
        self.assertEqual(value, [1.0])

    def test_central_difference_uses_both_sides(self):
        model = _FakeFold()
        result = estimate_implicit_mle(
            model,
            ["AC"],
            self.objective,
            perturb=0.0,
            nu=0.2,
            alignment="aligned",
            normalization="none",
            estimator_scale="divide_nu",
            difference_mode="central",
        )
        result.loss.sum().backward()
        torch.testing.assert_close(model.weight.grad, torch.tensor([5.0]))
        self.assertEqual(result.flip_fraction, [1.0])

    def test_multiple_samples_are_averaged_not_summed(self):
        model = _FakeFold()
        result = estimate_implicit_mle(
            model,
            ["AC"],
            self.objective,
            perturb=0.0,
            nu=0.2,
            alignment="aligned",
            normalization="none",
            estimator_scale="divide_nu",
            num_samples=3,
        )
        result.loss.sum().backward()
        torch.testing.assert_close(model.weight.grad, torch.tensor([5.0]))

    def test_adaptive_nu_increases_until_map_changes(self):
        model = _FakeFold()
        result = estimate_implicit_mle(
            model,
            ["AC"],
            self.objective,
            perturb=0.0,
            nu=0.01,
            alignment="aligned",
            normalization="none",
            estimator_scale="divide_nu",
            adaptive=AdaptiveStepConfig(
                enabled=True,
                minimum=0.01,
                maximum=1.0,
                flip_fraction_low=0.5,
                flip_fraction_high=1.0,
                factor=10.0,
                max_steps=3,
            ),
        )
        self.assertEqual(result.selected_nu, [0.1])
        self.assertEqual(result.flip_fraction, [1.0])

    def test_objective_parameter_gradient_can_pass_through(self):
        model = _FakeFold()
        objective_weight = torch.nn.Parameter(torch.tensor(2.0))

        def objective(paired):
            return torch.stack([objective_weight * pair[1:].mean() for pair in paired])

        result = estimate_implicit_mle(
            model,
            ["AC"],
            objective,
            perturb=0.0,
            nu=0.2,
            alignment="aligned",
            normalization="none",
            estimator_scale="divide_nu",
            propagate_value_gradient=True,
        )
        result.loss.sum().backward()
        torch.testing.assert_close(objective_weight.grad, torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
