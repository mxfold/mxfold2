"""Mechanism tests for the matched MW-retention pilot (no cluster needed)."""
import random
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.rnagym.mwu_retention_experiments import (
    AdditiveMWLoss, RetentionTrain, edit_distance, edit_pattern, separated,
)
from mxfold2.train import Train


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


class ToyStructural(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, seq, target, fname=None):
        return self.model.weight * 2


class ToyAuxiliary(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_imle_stats = {'flip_fraction': [.1]}

    def forward(self, seq, target, fname=None):
        torch.rand(3)  # Must not advance the structural RNG stream.
        random.random()
        return self.model.weight * 3


def toy_loss(model, coefficient):
    dataset = [('shape', 'ACGU', {'target': torch.zeros(5)})]
    loss = AdditiveMWLoss(ToyStructural(model), model, dataset, coefficient, 1234, -1)
    loss.auxiliary = ToyAuxiliary(model)
    return loss


def slow_distance(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        nxt = [i]
        for j, y in enumerate(b, 1):
            nxt.append(min(nxt[-1]+1, row[j]+1, row[j-1]+(x != y)))
        row = nxt
    return row[-1]


class RetentionTests(unittest.TestCase):
    def test_qsub_nonarray_and_array_task_index(self):
        root = Path(__file__).resolve().parents[1]
        for task, expected in ((None, '1'), ('undefined', '1'), ('', '1'), ('12', '12')):
            env = os.environ | {'PROJECT_ROOT': str(root), 'MWU_ROOT': '/tmp/mwu-shell-test', 'MWU_ACTION': 'preflight'}
            env.pop('SGE_TASK_ID', None)
            if task is not None:
                env['SGE_TASK_ID'] = task
            result = subprocess.run(['bash', '-c', 'exec() { printf "%s\\n" "$@"; }; source "$1"',
                                     'test-wrapper', str(root / 'scripts/rnagym/qsub_mwu_retention.sh')],
                                    env=env, check=True, text=True, capture_output=True)
            args = result.stdout.splitlines()
            self.assertEqual(args[args.index('--index') + 1], expected)

    def test_exact_global_edit_distance_and_threshold(self):
        rng = random.Random(17)
        for _ in range(200):
            a = ''.join(rng.choices('ACGU', k=rng.randrange(0, 180)))
            b = ''.join(rng.choices('ACGU', k=rng.randrange(0, 180)))
            self.assertEqual(edit_distance(edit_pattern(a), b), slow_distance(a, b))
        self.assertFalse(separated('AAAAC', [edit_pattern('AAAAA')]))
        self.assertTrue(separated('AAACC', [edit_pattern('AAAAA')]))

    def test_additive_gradient_and_rng_isolation(self):
        model = ToyModel()
        loss = toy_loss(model, .3)
        torch.manual_seed(7)
        random.seed(7)
        torch_state, python_state = torch.get_rng_state(), random.getstate()
        value = loss(['ACGU'], [torch.zeros(5)])
        value.backward()
        self.assertAlmostEqual(float(model.weight.grad), 2.9, places=5)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(python_state, random.getstate())
        self.assertEqual(loss.aux_calls, 1)

    def test_same_structural_exposures_and_optimizer_updates(self):
        for coefficient in (0., .1, .3):
            model = ToyModel()
            worker = RetentionTrain.__new__(RetentionTrain)
            Train.__init__(worker)
            worker.entry = {'coefficient': coefficient}
            worker.smoke = True
            worker.disable_progress_bar = True
            worker.use_wandb = False
            worker.combined = toy_loss(model, coefficient)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            updates = []
            optimizer.register_step_post_hook(lambda *args: updates.append(1))
            dataset = [('a', 'ACGU', {'type': 'BPSEQ', 'target': torch.zeros(5), 'dataset_id': 0})] * 3
            with tempfile.TemporaryDirectory() as directory:
                worker.folder = Path(directory)
                worker.train(1, model=model, optimizer=optimizer, loss_fn={'BPSEQ': worker.combined},
                             data_loader=DataLoader(dataset, batch_size=1))
            self.assertEqual(len(updates), 3)
            self.assertEqual(worker.step, 3)
            self.assertEqual(worker.combined.calls, 3)
            self.assertEqual(worker.combined.aux_calls, 3 if coefficient else 0)
            self.assertAlmostEqual(float(model.weight), 1-3*.01*(2+coefficient*3), places=5)


if __name__ == '__main__':
    unittest.main()
