import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from mxfold2.train import Train


class ToyDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return 'toy', 'A', {'type': 'BPSEQ', 'target': torch.tensor([1])}


class ToyLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, seqs, targets, fname=None):
        return .5 * self.model.weight.square().sum()


class BudgetSamplingTest(unittest.TestCase):
    def test_final_partial_update_uses_sampler_length_not_dataset_length(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1)
        dataset = ToyDataset()
        sampler = RandomSampler(dataset, replacement=True, num_samples=5)
        loader = DataLoader(dataset, batch_size=1, sampler=sampler)
        trainer = Train()
        trainer.disable_progress_bar = True
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        with patch.object(optimizer, 'step', wraps=optimizer.step) as step:
            trainer.train(1, model, optimizer, ToyLoss(model), loader, grad_accum_steps=4)
        self.assertEqual(trainer.step, 5)
        self.assertEqual(step.call_count, 2)
        self.assertAlmostEqual(model.weight.item(), .81, places=6)


if __name__ == '__main__':
    unittest.main()
