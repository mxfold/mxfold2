import random
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from scripts.rnagym.baseline_b_retention import CHECKPOINTS, Engine, ParameterAnchor, entries
from scripts.rnagym.mwu_b_smallbudget import save_state
from tests.test_mwu_b_smallbudget import ToyLoss


def toy_engine(coefficient=0., anchor_coefficient=0.):
    torch.manual_seed(1234)
    random.seed(1234)
    engine = Engine.__new__(Engine)
    engine.entry = dict(name='toy', seed=1234, method='mw', coefficient=coefficient,
                        anchor_coefficient=anchor_coefficient, accumulation=1)
    engine.gpu = -1
    engine.model = nn.Linear(2, 1, bias=False)
    # The engine implements the production AdaBelief state workaround.
    from mxfold2.train import Train
    engine.optimizer = Train().build_optimizer('AdaBelief', engine.model, .0001, .01)
    engine.structural = ToyLoss(engine.model)
    engine.chemical = ToyLoss(engine.model, chemical=True)
    engine.anchor = ParameterAnchor(engine.model, set(engine.model.state_dict()))
    engine.structures = engine.shapes = [('rna'+str(i), 'ACGU', dict(target=torch.tensor([float(i)]))) for i in range(5)]
    engine.structure_generator = torch.Generator().manual_seed(13)
    engine.shape_generator = torch.Generator().manual_seed(19)
    engine.structure_order = torch.randperm(5, generator=engine.structure_generator)
    engine.structure_cursor = engine.updates = engine.draws = 0
    return engine


class RetentionTests(unittest.TestCase):
    def test_matched_controls_and_budget(self):
        arms = entries()
        self.assertEqual(len(arms), 27)
        self.assertEqual(len({a['name'] for a in arms}), 27)
        for arm in arms:
            controls = [a for a in arms if a['seed'] == arm['seed']
                        and a['preservation'] == arm['preservation'] and a['method'] == 'structure']
            self.assertEqual(len(controls), 1)
            self.assertEqual(controls[0]['lr'], arm['lr'])
            self.assertEqual(controls[0]['anchor_coefficient'], arm['anchor_coefficient'])
            self.assertEqual(arm['pair_budget'], CHECKPOINTS[-1])

    def test_anchor_excludes_infinity_and_missing_heads(self):
        model = nn.Module()
        model.weight = nn.Parameter(torch.tensor([2., float('-inf')]))
        model.extra = nn.Parameter(torch.tensor([100.]))
        anchor = ParameterAnchor(model, {'weight'})
        self.assertEqual(anchor.metadata()['elements'], 1)
        self.assertEqual(float(anchor.raw()), 0.)
        with torch.no_grad():
            model.weight[0] += 3.
            model.extra += 10.
        self.assertAlmostEqual(float(anchor.raw()), 4.5)
        loss = anchor.weighted(.1)
        loss.backward()
        torch.testing.assert_close(model.weight.grad, torch.tensor([.3, 0.]))
        self.assertIsNone(model.extra.grad)
        self.assertFalse(anchor.weighted(0.).requires_grad)

    def test_parity_and_without_replacement(self):
        values = []
        for diagnose in (False, True):
            engine = toy_engine()
            engine.advance(5, diagnose=diagnose)
            self.assertEqual(len(set(engine.structure_order.tolist())), 5)
            self.assertEqual(engine.structure_cursor, 5)
            values.append((engine.model.weight.detach().clone(), torch.get_rng_state()))
        for first, second in zip(*values):
            torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_exact_resume_with_anchor_and_sampler_boundary(self):
        # Snapshot anchor references are reconstructed from the same baseline.
        whole = toy_engine(.5, .1)
        whole.advance(8, diagnose=False)
        expected, rng = whole.model.weight.detach().clone(), torch.get_rng_state()
        with tempfile.TemporaryDirectory() as directory:
            partial = toy_engine(.5, .1)
            partial.advance(5, diagnose=True)
            checkpoint = Path(directory) / 'resume.pth'
            save_state(checkpoint, partial.snapshot())
            resumed = toy_engine(.5, .1)
            resumed.restore(checkpoint)
            resumed.advance(8, diagnose=True)
        torch.testing.assert_close(expected, resumed.model.weight, rtol=0, atol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual((resumed.draws, resumed.structure_cursor), (8, 3))


if __name__ == '__main__':
    unittest.main()
