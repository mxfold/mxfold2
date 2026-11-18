import itertools
import math
import unittest
from argparse import ArgumentParser

import torch

from mxfold2.loss.nonparametric_rank import (
    mann_whitney_rank_loss,
    paired_reactivity_auc,
)
from mxfold2.loss.shape_rank_loss import ShapeRankLoss
from mxfold2.train import Train


def brute_force_auc(target, paired):
    """Independent enumeration of the cross-group comparisons, including ties."""
    successes = []
    for i in range(len(target)):
        for j in range(len(target)):
            if paired[i] == 1 and paired[j] == 0:
                successes.append(float(target[i] < target[j]) + 0.5 * (target[i] == target[j]))
    return sum(successes) / len(successes)


class NonparametricRankTest(unittest.TestCase):
    def test_every_binary_split_matches_pairwise_auc_with_ties(self):
        target = torch.tensor([0.0, 0.0, 2.0, 1.0, 3.0])
        for states in itertools.product([0.0, 1.0], repeat=5):
            if sum(states) in (0, 5):
                continue
            with self.subTest(states=states):
                paired = torch.tensor(states)
                expected = brute_force_auc(target.tolist(), states)
                self.assertAlmostEqual(float(paired_reactivity_auc(target, paired)), expected, places=6)
                self.assertAlmostEqual(float(mann_whitney_rank_loss(target, paired)), 1 - expected, places=6)

    def test_perfect_reversed_and_collapsed_predictions(self):
        target = torch.tensor([0.0, 0.2, 0.8, 1.0])
        for states, expected in [([1, 1, 0, 0], 0.0), ([0, 0, 1, 1], 1.0),
                                 ([0, 0, 0, 0], 0.5), ([1, 1, 1, 1], 0.5)]:
            self.assertAlmostEqual(float(mann_whitney_rank_loss(target, torch.tensor(states))), expected)

    def test_invariant_to_strictly_monotone_reactivity_transform(self):
        target = torch.tensor([0.0, 0.2, 0.2, 0.8, 1.0])
        paired = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        torch.testing.assert_close(mann_whitney_rank_loss(target, paired),
                                   mann_whitney_rank_loss(torch.exp(target), paired))
        torch.testing.assert_close(paired_reactivity_auc(target, paired),
                                   paired_reactivity_auc(torch.exp(target), paired))

    def test_simultaneous_permutation_preserves_statistic(self):
        target = torch.tensor([0.0, 0.2, 0.2, 0.8, 1.0])
        paired = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        indices = torch.tensor([4, 0, 2, 1, 3])
        torch.testing.assert_close(mann_whitney_rank_loss(target, paired),
                                   mann_whitney_rank_loss(target[indices], paired[indices]))

    def test_collapsed_predictions_keep_finite_rank_directed_gradients(self):
        target = torch.tensor([0.0, 0.2, 0.8, 1.0])
        for state in [0.0, 1.0]:
            paired = torch.full((4,), state, requires_grad=True)
            loss = mann_whitney_rank_loss(target, paired)
            loss.backward()
            self.assertEqual(float(loss), 0.5)
            self.assertTrue(bool(torch.isfinite(paired.grad).all()))
            self.assertLess(float(paired.grad[0]), 0)
            self.assertGreater(float(paired.grad[-1]), 0)
            self.assertAlmostEqual(float(paired.grad.sum()), 0.0, places=6)
            self.assertTrue(math.isnan(float(paired_reactivity_auc(target, paired))))

    def test_observation_ties_have_zero_gradient(self):
        target = torch.full((4,), 0.2)
        paired = torch.tensor([1.0, 0.0, 0.0, 1.0], requires_grad=True)
        loss = mann_whitney_rank_loss(target, paired)
        loss.backward()
        self.assertEqual(float(loss), 0.5)
        torch.testing.assert_close(paired.grad, torch.zeros(4))
        self.assertEqual(float(paired_reactivity_auc(target, paired)), 0.5)

    def test_missing_dummy_and_nonfinite_observations_are_omitted(self):
        target = torch.tensor([-999.0, 0.0, float('nan'), 0.2, float('inf'),
                               float('-inf'), 0.8, -999.0, 1.0])
        paired = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True)
        loss = mann_whitney_rank_loss(target, paired)
        loss.backward()
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(paired_reactivity_auc(target, paired)), 1.0)
        torch.testing.assert_close(paired.grad[torch.tensor([0, 2, 4, 5, 7])], torch.zeros(5))

    def test_fewer_than_two_observations_are_uninformative(self):
        for target in [torch.tensor([-999.0, -999.0]), torch.tensor([-999.0, 0.2])]:
            paired = torch.zeros(2, requires_grad=True)
            loss = mann_whitney_rank_loss(target, paired)
            loss.backward()
            self.assertEqual(float(loss), 0)
            torch.testing.assert_close(paired.grad, torch.zeros(2))
            self.assertTrue(math.isnan(float(paired_reactivity_auc(target, paired))))

    def test_soft_extension_matches_distinct_position_weighted_comparisons(self):
        target = torch.tensor([0.0, 0.2, 0.2, 0.8, 1.0], dtype=torch.float64)
        paired = torch.tensor([0.95, 0.8, 0.35, 0.15, 0.05], dtype=torch.float64)
        numerator = denominator = 0.0
        for i in range(len(target)):
            for j in range(len(target)):
                if i == j:
                    continue
                weight = float(paired[i] * (1 - paired[j]))
                numerator += weight * (float(target[i] < target[j]) + 0.5 * float(target[i] == target[j]))
                denominator += weight
        expected = 0.5 - (numerator - 0.5 * denominator) / max(denominator, len(target) - 1)
        self.assertAlmostEqual(float(mann_whitney_rank_loss(target, paired)), expected, places=12)

    def test_soft_gradient_matches_numerical_derivative(self):
        target = torch.tensor([0.0, 0.2, 0.8, 1.0], dtype=torch.float64)
        paired = torch.tensor([0.8, 0.7, 0.3, 0.2], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda p: mann_whitney_rank_loss(target, p), (paired,)))

    def test_nearly_collapsed_soft_states_are_bounded_and_finite(self):
        target = torch.linspace(0, 1, 100)
        for offset in [0.0, 1e-6, 0.5, 1 - 1e-6, 1.0]:
            paired = torch.full((100,), offset, requires_grad=True)
            loss = mann_whitney_rank_loss(target, paired)
            loss.backward()
            self.assertGreaterEqual(float(loss), 0)
            self.assertLessEqual(float(loss), 1)
            self.assertTrue(bool(torch.isfinite(paired.grad).all()))

    def test_bad_shapes_and_pairedness_are_rejected(self):
        for target, paired in [(torch.ones(2, 1), torch.zeros(2)),
                               (torch.ones(2), torch.zeros(3)),
                               (torch.ones(2), torch.tensor([0.0, float('nan')])),
                               (torch.ones(2), torch.tensor([0.0, -0.1])),
                               (torch.ones(2), torch.tensor([0.0, 1.1]))]:
            with self.assertRaises(ValueError):
                mann_whitney_rank_loss(target, paired)
        with self.assertRaisesRegex(ValueError, 'binary'):
            paired_reactivity_auc(torch.tensor([0.0, 1.0]), torch.tensor([0.2, 0.8]))


class _RankOracle(torch.nn.Module):
    """Exact toy MAP over a good split, a reversed split and an empty structure."""

    def __init__(self, initial):
        super().__init__()
        weights = torch.zeros(3)
        weights[initial] = 0.05
        self.weights = torch.nn.Parameter(weights)
        self.register_buffer('masks', torch.tensor([[1., 1., 0., 0.],
                                                  [0., 0., 1., 1.],
                                                  [0., 0., 0., 0.]]))

    def forward(self, seq, param=None, pseudoenergy=None, return_param=False,
                return_count=False, perturb=0.0):
        del return_count, perturb
        if len(seq) != 1:
            raise AssertionError('Toy oracle expects batch size one')
        if param is None:
            param = [{'score_candidates': self.weights}]
        energies = self.weights.new_zeros(4) if pseudoenergy is None else pseudoenergy[0]
        selection_scores = param[0]['score_candidates'].detach() - self.masks @ energies
        selected = int(selection_scores.argmax())
        counts = torch.nn.functional.one_hot(torch.tensor(selected), 3).to(self.weights)
        param[0]['count_candidates'] = counts
        score = (param[0]['score_candidates'] * counts).sum().reshape(1)
        structures = ['()..', '..()', '....']
        pairs = [[0, 2, 1, 0, 0], [0, 0, 0, 4, 3], [0, 0, 0, 0, 0]]
        if return_param:
            return score, [structures[selected]], [pairs[selected]], param, param
        return score, [structures[selected]], [pairs[selected]]


class RankObjectiveIntegrationTest(unittest.TestCase):
    def test_imle_update_improves_reversed_and_empty_predictions(self):
        for initial in [1, 2]:
            for aggregation in ['independent', 'mean_paired']:
                with self.subTest(initial=initial, aggregation=aggregation):
                    model = _RankOracle(initial)
                    loss_fn = ShapeRankLoss(model, objective='mann_whitney', nu=0.1,
                                            normalization='rms', difference_mode='central',
                                            sample_aggregation=aggregation, num_samples=3)
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
                    target = torch.tensor([-999.0, 0.0, 0.2, 0.8, 1.0])
                    before = loss_fn(['GCGC'], [target])
                    before.backward()
                    self.assertTrue(bool(torch.isfinite(model.weights.grad).all()))
                    self.assertGreater(float(model.weights.grad.abs().sum()), 0)
                    optimizer.step()
                    after = loss_fn(['GCGC'], [target])
                    self.assertLess(float(after), float(before))
                    self.assertEqual(float(after), 0.0)

    def test_cli_and_factory_select_new_objective_without_changing_default(self):
        parser = ArgumentParser()
        Train.add_args(parser.add_subparsers())
        for objective in ['pairwise_logistic', 'mann_whitney']:
            command = ['train', '--shape-loss-func', 'shape_rank', 'empty.lst']
            if objective == 'mann_whitney':
                command.extend(['--shape-rank-objective', objective])
            args = parser.parse_args(command)
            loss_fn = Train().build_shape_loss_function('shape_rank', torch.nn.Linear(1, 1), args)
            self.assertEqual(loss_fn.objective, objective)

    def test_logistic_options_are_not_silently_ignored(self):
        for options in [{'margin': 0.1}, {'rank_delta': 0.002},
                        {'teacher_temperature': 0.03}, {'teacher_normalization': 'percentile'},
                        {'weighting': 'gap'}, {'clip_targets': True}]:
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'raw ranks'):
                ShapeRankLoss(torch.nn.Linear(1, 1), objective='mann_whitney', **options)
        with self.assertRaisesRegex(ValueError, 'objective'):
            ShapeRankLoss(torch.nn.Linear(1, 1), objective='unknown')


if __name__ == '__main__':
    unittest.main()
