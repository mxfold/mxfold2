"""Gradient diagnostics must not change the training they measure."""
import random
import unittest

import torch
from torch import nn

from mxfold2.loss.fy_loss import FenchelYoungLoss
from scripts.rnagym.mwu_weak_experiments import (
    DecomposedFYLoss, coefficient_at, diagnostic_window, gradient_statistics,
)


class ToyFold(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([.3, -.2, .8]))

    def forward(self, seq, param=None, constraint=None, return_param=False, perturb=0, **kwargs):
        scores = self.weights * torch.rand(3) if param is None else param
        value = (scores * torch.tensor([1., 2., 3.] if constraint is None else [2., -1., 1.])).sum().view(1)
        result = (value, ['.' * len(seq[0])], [torch.zeros(len(seq[0])+1)])
        return (*result, None, scores) if return_param else result


class ToyTurner(nn.Module):
    def forward(self, seq, **kwargs):
        return torch.tensor([-.7]), ['.' * len(seq[0])], [None]


def loss_instance(kind, model, score_weight):
    loss = kind.__new__(kind)
    nn.Module.__init__(loss)
    loss.model, loss.turner = model, ToyTurner()
    loss.perturb, loss.sl_weight = .75, score_weight
    loss.l1_weight, loss.l2_weight = 0., .01
    return loss


class WeakMWTests(unittest.TestCase):
    def test_exact_fy_values_gradients_and_rng(self):
        for weight in (0., .125):
            results = []
            for kind in (FenchelYoungLoss, DecomposedFYLoss):
                model = ToyFold()
                loss = loss_instance(kind, model, weight)
                torch.manual_seed(19)
                value = loss(['ACGUACGU'], [torch.zeros(9)], fname=['test'])
                value.sum().backward()
                results.append((value.detach(), model.weights.grad.clone(), torch.get_rng_state()))
                if kind is DecomposedFYLoss:
                    torch.testing.assert_close(value.sum(), loss.parts['fy']+loss.parts['thermo'], rtol=0, atol=0)
            for plain, split in zip(results[0], results[1]):
                torch.testing.assert_close(plain, split, rtol=0, atol=0)

    def test_diagnostics_measure_conflict_without_changing_gradients(self):
        model = nn.Module()
        model.net = nn.Module()
        model.net.encoder = nn.Linear(2, 1, bias=False)
        model.net.fc_paired = nn.Linear(2, 1, bias=False)
        # An unused group is deliberately present.
        model.net.embedding = nn.Linear(2, 1, bias=False)
        fy = (model.net.encoder.weight * torch.tensor([[3., 4.]])).sum()
        thermo = (model.net.fc_paired.weight * torch.tensor([[0., 2.]])).sum()
        mw = -fy + 2*thermo
        torch_state, python_state = torch.get_rng_state(), random.getstate()
        rows = {row['group']: row for row in gradient_statistics(model, fy, thermo, mw, .01)}
        self.assertEqual(rows['encoder']['cos_fy_mw'], -1.)
        self.assertEqual(rows['fc_paired']['cos_thermo_mw'], 1.)
        self.assertEqual(rows['embedding']['cos_structure_mw'], '')
        self.assertAlmostEqual(rows['all']['structure_norm'], 29**.5)
        self.assertAlmostEqual(rows['all']['mw_norm'], 41**.5)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        (fy+thermo+.01*mw).backward()
        torch.testing.assert_close(model.net.encoder.weight.grad, torch.tensor([[2.97, 3.96]]))
        torch.testing.assert_close(model.net.fc_paired.weight.grad, torch.tensor([[0., 2.04]]))
        self.assertIsNone(model.net.embedding.weight.grad)

    def test_decay_endpoints_and_epoch_boundaries(self):
        self.assertEqual(coefficient_at(.01, 5, 1, 3166), .01)
        self.assertEqual(coefficient_at(.01, 5, 15830, 3166), 0.)
        self.assertEqual(coefficient_at(.01, 5, 15831, 3166), 0.)
        self.assertEqual(coefficient_at(.003, 0, 63320, 3166), .003)
        values = [coefficient_at(.01, 5, step, 8) for step in range(1, 49)]
        self.assertTrue(all(a >= b >= 0 for a, b in zip(values, values[1:])))
        self.assertGreater(values[7], values[8])
        self.assertEqual(sum(x > 0 for x in values), 39)
        self.assertEqual(diagnostic_window(100), 100)
        self.assertEqual(diagnostic_window(115), 100)
        self.assertIsNone(diagnostic_window(116))


if __name__ == '__main__':
    unittest.main()
