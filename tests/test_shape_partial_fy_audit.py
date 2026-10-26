"""Independent enumeration and numerical-gradient audit of partial MAP FY.

The oracle enumerates canonical, noncrossing structures, and filters them by
the observed unpaired positions without calling the loss's constraint code.
Exact forced-structure scores come from the real C++ nearest-neighbor backend.
"""
from functools import lru_cache
import unittest

import torch

from mxfold2.fold.contrafold import CONTRAfold
from mxfold2.loss.fy_loss import FenchelYoungLoss
from mxfold2.loss.shape_partial_fy_loss import ShapePartialFenchelYoungLoss


def enumerate_structures(sequence):
    canonical = {"AU", "UA", "GC", "CG", "GU", "UG"}

    @lru_cache(None)
    def interval(left, right):
        if left > right:
            return ((),)
        result = list(interval(left + 1, right))
        for partner in range(left + 4, right + 1):
            if sequence[left - 1] + sequence[partner - 1] not in canonical:
                continue
            for inside in interval(left + 1, partner - 1):
                for outside in interval(partner + 1, right):
                    result.append(((left, partner),) + inside + outside)
        return tuple(result)

    vectors = []
    for pairs in interval(1, len(sequence)):
        vector = torch.zeros(len(sequence) + 1, dtype=torch.long)
        for i, j in pairs:
            vector[i], vector[j] = j, i
        vectors.append(vector)
    return vectors


def randomized_contrafold(seed):
    generator = torch.Generator().manual_seed(seed)
    model = CONTRAfold()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * .15)
        model.score_base_pair.add_(1.)
    return model


def enumerated_scores(model, sequence):
    vectors = enumerate_structures(sequence)
    parameters = model.make_param([sequence])
    scores = []
    for vector in vectors:
        score, _, reconstructed = model([sequence], param=parameters, constraint=[vector])
        if reconstructed[0] != vector.tolist():
            raise AssertionError("Backend altered an enumerated legal structure")
        scores.append(score[0])
    return vectors, torch.stack(scores)


class PartialFYIndependentAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_every_unpaired_mask_matches_exhaustive_oracle(self):
        sequence = "GCGAAACGC"
        model = randomized_contrafold(73)
        loss_fn = ShapePartialFenchelYoungLoss(model, unpaired_threshold=.8)
        with torch.no_grad():
            vectors, scores = enumerated_scores(model, sequence)
            vectors = torch.stack(vectors)
            for mask in range(1 << len(sequence)):
                high = torch.tensor([bool(mask & (1 << i)) for i in range(len(sequence))])
                target = torch.where(high, 1., .1)
                feasible = (vectors[:, 1:][:, high] == 0).all(dim=1)
                expected = (scores.max() - scores[feasible].max()) / len(sequence)
                actual = loss_fn([sequence], [target])[0]
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
                self.assertGreaterEqual(float(actual), -2e-6)

    def test_enumerated_feature_gradients_match_for_every_parameter(self):
        for sequence, seed in [("GCGAAACGC", 73), ("GGGAAACCC", 19), ("GUGCAAAUGCAC", 11)]:
            model = randomized_contrafold(seed)
            parameters = tuple(model.parameters())
            vectors, scores = enumerated_scores(model, sequence)
            vectors = torch.stack(vectors)
            loss_fn = ShapePartialFenchelYoungLoss(model)
            for positions in [(1,), (2,), (1, 3), tuple(range(1, len(sequence) + 1))]:
                target = torch.zeros(len(sequence))
                target[torch.tensor(positions) - 1] = 1.
                feasible = (vectors[:, positions] == 0).all(dim=1)
                oracle = (scores.max() - scores[feasible].max()) / len(sequence)
                expected = torch.autograd.grad(oracle, parameters, retain_graph=True)
                actual_loss = loss_fn([sequence], [target]).sum()
                actual = torch.autograd.grad(actual_loss, parameters)
                torch.testing.assert_close(actual_loss, oracle, atol=2e-6, rtol=2e-5)
                for gradient, reference in zip(actual, expected):
                    torch.testing.assert_close(gradient, reference, atol=2e-6, rtol=2e-5)

    def test_contrafold_directional_derivative_matches_finite_difference(self):
        model = randomized_contrafold(73)
        sequence = "GCGAAACGC"
        target = torch.tensor([1., 1., 1., 0., 0., 0., 0., 0., 0.])
        loss_fn = ShapePartialFenchelYoungLoss(model)
        loss = loss_fn([sequence], [target]).sum()
        gradient, = torch.autograd.grad(loss, model.score_base_pair)
        self.assertGreater(float(gradient.norm()), 0.)
        direction = gradient / gradient.norm()
        original = model.score_base_pair.detach().clone()
        eps, values = 1e-3, []
        with torch.no_grad():
            for sign in (1., -1.):
                model.score_base_pair.copy_(original + sign * eps * direction)
                values.append(float(loss_fn([sequence], [target])))
            model.score_base_pair.copy_(original)
        numerical = (values[0] - values[1]) / (2 * eps)
        self.assertAlmostEqual(numerical, float((gradient * direction).sum()), delta=3e-4)

    def test_all_high_reactivity_reduces_to_fully_unpaired_fy(self):
        model = randomized_contrafold(73)
        sequences = ["GCGAAACGC", "GGGAAACCC"]
        partial = ShapePartialFenchelYoungLoss(model, perturb=0.)
        complete = FenchelYoungLoss(model, perturb=0., sl_weight=0.)
        actual = partial(sequences, [torch.ones(len(s)) for s in sequences])
        # Ordinary FY's diagnostic logging currently assumes batch size one.
        expected = torch.cat([
            complete([s], [torch.zeros(len(s) + 1, dtype=torch.long)])
            for s in sequences
        ])
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_real_backend_batch_and_single_calls_agree(self):
        model = randomized_contrafold(73)
        sequences = ["GCGAAACGC", "GGGAAACCC", "AAAAAA"]
        targets = [torch.ones(9), torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0.]), torch.full((6,), -999.)]
        loss_fn = ShapePartialFenchelYoungLoss(model)
        batch = loss_fn(sequences, targets)
        singles = torch.cat([loss_fn([s], [r]) for s, r in zip(sequences, targets)])
        torch.testing.assert_close(batch, singles, atol=0., rtol=0.)


class PartialFYBaselineAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.rnagym import mwu_retention_experiments as common
        from scripts.rnagym.mwu_weak_experiments import native_lstm

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.model = native_lstm(common.load_model(common.BASELINE, -1))

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_actual_mix_weights_and_directional_gradient(self):
        model = self.model.eval()
        self.assertEqual(model.score_weight_positional, model.count_weight_positional)
        self.assertEqual(model.score_weight_turner, model.count_weight_turner)
        sequence = "GCGAAAACGC"
        target = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        loss_fn = ShapePartialFenchelYoungLoss(model)
        loss = loss_fn([sequence], [target]).sum()
        candidates = [(n, p) for n, p in model.named_parameters() if n.startswith("zuker.") and p.numel() <= 1024]
        gradients = torch.autograd.grad(loss, [p for _, p in candidates], allow_unused=True)
        options = [(float(g.norm()), p, g) for (_, p), g in zip(candidates, gradients) if g is not None]
        norm, parameter, gradient = max(options, key=lambda row: row[0])
        self.assertGreater(norm, 0.)
        direction, original = gradient / norm, parameter.detach().clone()
        eps, values = 2e-3, []
        with torch.no_grad():
            for sign in (1., -1.):
                parameter.copy_(original + sign * eps * direction)
                values.append(float(loss_fn([sequence], [target])))
            parameter.copy_(original)
        self.assertAlmostEqual((values[0] - values[1]) / (2 * eps), norm, delta=2e-3)

    def test_missing_data_gives_exact_zero_with_real_dropout_and_perturbation(self):
        model = self.model.train()
        model.zero_grad(set_to_none=True)
        loss = ShapePartialFenchelYoungLoss(model, perturb=.1)(
            ["GCGAAAACGC"], [torch.full((11,), -999.)])
        loss.sum().backward()
        torch.testing.assert_close(loss, torch.zeros(1), atol=0., rtol=0.)
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.count_nonzero(g) == 0 for g in gradients))
        model.eval()

    def test_neural_parameter_generation_happens_once_for_both_maxima(self):
        from unittest.mock import patch

        model = self.model.train()
        target = torch.tensor([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        with patch.object(model, "make_param", wraps=model.make_param) as generated:
            loss = ShapePartialFenchelYoungLoss(model, perturb=.1)(["GCGAAAACGC"], [target])
        self.assertEqual(generated.call_count, 1)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreaterEqual(float(loss.detach()), -2e-6)
        model.eval()


if __name__ == "__main__":
    unittest.main()
