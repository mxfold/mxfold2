import unittest
from argparse import ArgumentParser

import torch
import torch.nn as nn

from mxfold2.loss.shape_partial_fy_loss import ShapePartialFenchelYoungLoss


class TwoStructureFold(nn.Module):
    """Exact MAP oracle with two paired structures and an unpaired fallback."""

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([2.0, 1.0]))
        self.make_param_calls = 0
        self.calls = []

    def make_param(self, seq, perturb=0.0):
        self.make_param_calls += 1
        base = [{"score_candidates": self.weights} for _ in seq]
        if perturb > 0.0:
            perturbed = [
                {"score_candidates": self.weights + perturb * self.weights.new_tensor([1.0, 2.0])}
                for _ in seq
            ]
            return perturbed, base
        return base

    def forward(self, seq, param=None, constraint=None):
        if param is None:
            raise AssertionError("partial FY must reuse generated parameters")
        self.calls.append((list(seq), list(param), constraint))
        scores, pairs, dot_brackets = [], [], []
        for i, s in enumerate(seq):
            candidates = torch.zeros((3, len(s) + 1), dtype=torch.long)
            candidates[0, 1], candidates[0, len(s)] = len(s), 1
            candidates[1, 2], candidates[1, len(s) - 1] = len(s) - 1, 2
            valid = torch.ones(3, dtype=torch.bool)
            if constraint is not None:
                unpaired = constraint[i].cpu() == 0
                valid = (candidates[:, unpaired] == 0).all(dim=1)
            values = torch.cat([
                param[i]["score_candidates"],
                param[i]["score_candidates"].sum().reshape(1) * 0.0,
            ])
            chosen = values.detach().masked_fill(~valid, -torch.inf).argmax().item()
            scores.append(values[chosen])
            pairs.append(candidates[chosen])
            dot_brackets.append("." * len(s))
        return torch.stack(scores), dot_brackets, pairs


class ShapePartialFenchelYoungLossTest(unittest.TestCase):
    def test_dummy_is_ignored_and_only_high_finite_measurements_are_unpaired(self):
        loss_fn = ShapePartialFenchelYoungLoss(TwoStructureFold())
        target = torch.tensor([99.0, 0.8, 0.0, -999.0, float("nan"), float("inf"), 1.2])
        original = target.clone()

        constraint = loss_fn._make_constraint("ACGUAC", target)

        torch.testing.assert_close(constraint, torch.tensor([-1, 0, -1, -1, -1, -1, 0]))
        torch.testing.assert_close(target, original, equal_nan=True)
        self.assertFalse(constraint.requires_grad)

    def test_targets_with_and_without_dummy_produce_same_constraints(self):
        loss_fn = ShapePartialFenchelYoungLoss(TwoStructureFold())
        values = torch.tensor([0.9, 0.1, -999.0, 0.5, 0.8, 0.0])
        torch.testing.assert_close(
            loss_fn._make_constraint("ACGUAC", values),
            loss_fn._make_constraint("ACGUAC", torch.cat([torch.tensor([-999.0]), values])),
        )

    def test_threshold_changes_only_unpaired_constraints(self):
        loss_fn = ShapePartialFenchelYoungLoss(TwoStructureFold(), unpaired_threshold=1.0)
        constraint = loss_fn._make_constraint("ACGUAC", torch.tensor([0.9, 1.0, 2.0, 0.0, -1.0, 0.1]))
        torch.testing.assert_close(constraint, torch.tensor([-1, -1, 0, 0, -1, -1, -1]))

    def test_coupled_perturbations_and_dropout_parameters_are_reused(self):
        model = TwoStructureFold()
        loss_fn = ShapePartialFenchelYoungLoss(model, perturb=0.2)

        loss = loss_fn(["GAAACC"], [torch.tensor([0.9, 0.0, 0.0, 0.0, 0.0, 0.0])])

        # The two maxima score 2.2 and 1.4 under the same perturbation.
        torch.testing.assert_close(loss, torch.tensor([0.8 / 6]))
        self.assertEqual(model.make_param_calls, 1)
        self.assertEqual(len(model.calls), 2)
        self.assertIs(model.calls[0][1][0], model.calls[1][1][0])

    def test_count_difference_gradient_discourages_violation_and_rewards_valid_partner(self):
        model = TwoStructureFold()
        loss_fn = ShapePartialFenchelYoungLoss(model)
        target = torch.tensor([0.9, 0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
        loss = loss_fn(["GAAACC"], [target])
        loss.sum().backward()

        torch.testing.assert_close(loss, torch.tensor([1.0 / 6]))
        torch.testing.assert_close(model.weights.grad, torch.tensor([1.0 / 6, -1.0 / 6]))
        self.assertIsNone(target.grad)
        self.assertTrue(torch.isfinite(model.weights.grad).all())

        before = model.weights.detach().clone()
        with torch.no_grad():
            model.weights -= 0.5 * model.weights.grad
        self.assertLess(float(model.weights[0]), float(before[0]))
        self.assertGreater(float(model.weights[1]), float(before[1]))
        self.assertLess(float(loss_fn(["GAAACC"], [target])), float(loss))

    def test_missing_or_low_reactivity_gives_exact_zero_even_with_perturbation(self):
        model = TwoStructureFold()
        loss_fn = ShapePartialFenchelYoungLoss(model, perturb=0.2)
        loss = loss_fn(["GAAACC"], [torch.tensor([-999.0, float("nan"), float("inf"), 0.0, 0.4, 0.79])])
        loss.sum().backward()

        torch.testing.assert_close(loss, torch.zeros(1), atol=0.0, rtol=0.0)
        torch.testing.assert_close(model.weights.grad, torch.zeros(2), atol=0.0, rtol=0.0)
        self.assertEqual(model.make_param_calls, 1)
        self.assertEqual(len(model.calls), 1)

    def test_satisfied_constraints_give_zero_score_gap_and_gradient(self):
        model = TwoStructureFold()
        loss_fn = ShapePartialFenchelYoungLoss(model)
        # Position 3 is unpaired in every candidate; the MAP remains unchanged.
        loss = loss_fn(["GAAACC"], [torch.tensor([0.0, 0.0, 0.9, 0.0, 0.0, 0.0])])
        loss.sum().backward()
        torch.testing.assert_close(loss, torch.zeros(1))
        torch.testing.assert_close(model.weights.grad, torch.zeros(2))

    def test_batch_uses_only_active_sequences_and_normalizes_by_their_length(self):
        model = TwoStructureFold()
        loss_fn = ShapePartialFenchelYoungLoss(model)
        seq = ["GAAACC", "GAAAAC", "GAAAAACC"]
        targets = [torch.zeros(6), torch.tensor([0.9, 0, 0, 0, 0, 0]), torch.tensor([0.9, 0, 0, 0, 0, 0, 0, 0])]

        loss = loss_fn(seq, targets)
        loss.sum().backward()

        torch.testing.assert_close(loss, torch.tensor([0.0, 1 / 6, 1 / 8]))
        torch.testing.assert_close(model.weights.grad, torch.tensor([1 / 6 + 1 / 8, -1 / 6 - 1 / 8]))
        self.assertEqual(model.calls[1][0], seq[1:])
        self.assertIs(model.calls[0][1][1], model.calls[1][1][0])
        self.assertIs(model.calls[0][1][2], model.calls[1][1][1])

    def test_bad_shape_dimensions_and_batch_sizes_are_rejected_before_folding(self):
        model = TwoStructureFold()
        loss_fn = ShapePartialFenchelYoungLoss(model)
        with self.assertRaisesRegex(ValueError, "SHAPE target length"):
            loss_fn(["GAAACC"], [torch.ones(4)])
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            loss_fn(["GAAACC"], [torch.ones((1, 6))])
        with self.assertRaisesRegex(ValueError, "batch size mismatch"):
            loss_fn(["GAAACC"], [])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            loss_fn([], [])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            loss_fn([""], [torch.empty(0)])
        self.assertEqual(model.make_param_calls, 0)

    def test_invalid_threshold_perturbation_and_full_reference_regularizer_are_rejected(self):
        for threshold in (-0.1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, "threshold"):
                ShapePartialFenchelYoungLoss(TwoStructureFold(), unpaired_threshold=threshold)
        for perturb in (-0.1, float("nan"), float("inf")):
            with self.subTest(perturb=perturb), self.assertRaisesRegex(ValueError, "perturb"):
                ShapePartialFenchelYoungLoss(TwoStructureFold(), perturb=perturb)
        with self.assertRaisesRegex(ValueError, "score-loss-weight 0"):
            ShapePartialFenchelYoungLoss(TwoStructureFold(), sl_weight=0.1)

    def test_nonfinite_fold_score_is_rejected(self):
        model = TwoStructureFold()
        with torch.no_grad():
            model.weights[0] = torch.nan
        with self.assertRaisesRegex(RuntimeError, "nonfinite unconstrained"):
            ShapePartialFenchelYoungLoss(model)(["GAAACC"], [torch.ones(6)])

    def test_cli_builds_partial_fy_with_configured_threshold(self):
        from mxfold2.train import Train

        parser = ArgumentParser()
        Train.add_args(parser.add_subparsers())
        args = parser.parse_args([
            "train", "example.lst", "--shape-loss-func", "shape_partial_fy",
            "--shape-unpaired-threshold", "1.2", "--score-loss-weight", "0",
        ])
        loss_fn = Train().build_shape_loss_function(args.shape_loss_func, TwoStructureFold(), args)
        self.assertIsInstance(loss_fn, ShapePartialFenchelYoungLoss)
        self.assertEqual(loss_fn.unpaired_threshold, 1.2)

    def test_cpp_backend_obeys_partial_constraints_and_produces_count_gradient(self):
        from mxfold2.fold.contrafold import CONTRAfold

        model = CONTRAfold()
        with torch.no_grad():
            model.score_base_pair.fill_(2.0)
        loss_fn = ShapePartialFenchelYoungLoss(model)
        loss = loss_fn(["GAAAC"], [torch.tensor([-999.0, 0.9, 0.0, 0.0, 0.0, 0.0])])
        loss.sum().backward()

        # The full MAP contains one GC pair (score 2). Forcing only its G
        # unpaired leaves all other positions free, and yields score zero.
        torch.testing.assert_close(loss, torch.tensor([2.0 / 5]))
        torch.testing.assert_close(model.score_base_pair.grad.sum(), torch.tensor(1.0 / 5))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
