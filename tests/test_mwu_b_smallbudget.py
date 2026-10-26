import unittest
import random
from pathlib import Path
import tempfile
import torch
from torch import nn
from scripts.rnagym.mwu_b_smallbudget import Engine, entries, choose, save_state, UPDATES, ACCUM8_UPDATES


class ToyLoss(nn.Module):
    def __init__(self, model, chemical=False):
        super().__init__()
        self.model, self.chemical = model, chemical

    def forward(self, seq, targets, fname=None):
        # Stochastic forwards verify auxiliary RNG isolation as well as updates.
        direction = torch.tensor([1., -1.] if self.chemical else [1., 2.])
        value = .02*((self.model.weight*direction*torch.rand(2)).sum()-targets[0].sum()).square()
        if self.chemical:
            value = value*.2
        self.parts = dict(fy=value, thermo=torch.zeros_like(value))
        return value.reshape(1)


def toy_engine(weight, accumulation):
    torch.manual_seed(1234)
    random.seed(1234)
    engine = Engine.__new__(Engine)
    engine.entry = dict(name='toy', seed=1234, method='mw', coefficient=weight, accumulation=accumulation)
    engine.gpu = -1
    engine.model = nn.Linear(2, 1, bias=False)
    engine.optimizer = torch.optim.AdamW(engine.model.parameters(), lr=1e-4, weight_decay=1e-4)
    engine.structural = ToyLoss(engine.model)
    engine.chemical = ToyLoss(engine.model, chemical=True)
    engine.structures = engine.shapes = [('rna', 'ACGU', dict(target=torch.tensor([float(i)]))) for i in range(3)]
    engine.structure_generator = torch.Generator().manual_seed(13)
    engine.shape_generator = torch.Generator().manual_seed(19)
    engine.updates = engine.draws = 0
    return engine


class BProtocolTests(unittest.TestCase):
    def test_budget_and_controls(self):
        arms = entries()
        self.assertEqual(len(arms), 30)
        self.assertEqual(len({a['name'] for a in arms}), 30)
        for arm in arms:
            self.assertEqual(arm['checkpoints'][-1]*arm['accumulation'], arm['pair_budget'])
            self.assertEqual(arm['pair_budget'], 2560)
            controls = [a for a in arms if a['method'] == 'structure' and a['lr'] == arm['lr']
                        and a['seed'] == arm['seed'] and a['accumulation'] == arm['accumulation']]
            self.assertEqual(len(controls), 1)
        self.assertEqual(UPDATES[0], 0)
        self.assertEqual(ACCUM8_UPDATES[0], 0)

    def test_selection_earliest_tie_and_baseline(self):
        curve = [dict(updates=300, F1=.5), dict(updates=0, F1=.5), dict(updates=100, F1=.4)]
        self.assertEqual(choose(curve)['updates'], 0)
        curve[0]['F1'] = .6
        self.assertEqual(choose(curve)['updates'], 300)
        with self.assertRaises(FloatingPointError):
            choose([dict(updates=0, F1=float('nan'))])

    def test_engine_diagnostic_parity_and_exposure(self):
        results = []
        for diagnose in (False, True):
            engine = toy_engine(0., 8)
            engine.advance(4, diagnose=diagnose)
            self.assertEqual((engine.updates, engine.draws), (4, 32))
            results.append((engine.model.weight.detach().clone(), torch.get_rng_state()))
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_engine_exact_resume_and_auxiliary_effect(self):
        for accumulation in (1, 8):
            whole = toy_engine(.5, accumulation)
            whole.advance(4, diagnose=False)
            state, rng = whole.model.weight.detach().clone(), torch.get_rng_state()
            with tempfile.TemporaryDirectory() as directory:
                first = toy_engine(.5, accumulation)
                first.advance(2, diagnose=True)
                save_state(Path(directory)/'resume.pth', first.snapshot())
                resumed = toy_engine(.5, accumulation)
                resumed.restore(Path(directory)/'resume.pth')
                resumed.advance(4, diagnose=True)
            torch.testing.assert_close(state, resumed.model.weight, rtol=0, atol=0)
            torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
            self.assertEqual(resumed.draws, 4*accumulation)
            zero = toy_engine(0., accumulation)
            zero.advance(4, diagnose=False)
            self.assertFalse(torch.equal(state, zero.model.weight))


if __name__ == '__main__':
    unittest.main()
