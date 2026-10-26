from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import torch
from torch import nn

from scripts.rnagym.mxfold2_fixed_teacher import (FixedPseudoFY, PackedTeachers, TEACHERS,
                                                entries, pseudoenergy, validate_pairs)


def fixture(path):
    with Path(path).open('wb') as stream:
        np.savez_compressed(stream, names=np.asarray(['rna0', 'rna1']), sequences=np.asarray(['ACGU', 'AAAA']),
            offsets=np.asarray([0, 5, 10]), reactivities=np.asarray([-999., .1, .2, .3, .4]*2, dtype=np.float32),
            baseline_pseudo=np.asarray([0, 4, 3, 2, 1, 0, 0, 0, 0, 0], dtype=np.int32),
            baseline_assisted_pseudo=np.zeros(10, dtype=np.int32))


class TeacherTests(unittest.TestCase):
    def test_budget_and_matching(self):
        arms = entries()
        self.assertEqual(len(arms), 18)
        self.assertEqual(len({a['name'] for a in arms}), 18)
        for arm in arms:
            counterpart = [a for a in arms if a['seed'] == arm['seed']
                           and a['preservation'] == arm['preservation'] and a['method'] != arm['method']]
            self.assertEqual(len(counterpart), 1)
            for key in ('lr', 'anchor_coefficient', 'coefficient', 'checkpoints', 'pair_budget'):
                self.assertEqual(arm[key], counterpart[0][key])

    def test_energy_removes_dummy_and_masks_missing(self):
        raw = torch.tensor([-999., 0., 1., -999., float('nan'), -1., -.5])
        energy = pseudoenergy(raw, 6)
        expected = torch.tensor([-.8, 2.6*np.log(2)-.8, 0., 0., 0., 2.6*np.log(.5)-.8], dtype=raw.dtype)
        torch.testing.assert_close(energy, expected)
        with self.assertRaises(ValueError):
            pseudoenergy(raw[1:], 6)
        with self.assertRaises(FloatingPointError):
            pseudoenergy(torch.tensor([-999., float('inf')]), 1)

    def test_partner_validation(self):
        validate_pairs([0, 4, 3, 2, 1], 4)
        validate_pairs([0, 0, 0], 2)
        for pairs in ([0, 1, 0], [0, 2, 0], [0, 3, 4, 1, 2]):
            with self.assertRaises(ValueError):
                validate_pairs(pairs, len(pairs)-1)

    def test_packed_roundtrip_and_fixed_loss_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'teachers.npz'
            fixture(path)
            packed = PackedTeachers(path)
            self.assertEqual(len(packed), 2)
            self.assertEqual(packed[0][1], 'ACGU')
            torch.testing.assert_close(packed.target(TEACHERS[0], 0), torch.tensor([0, 4, 3, 2, 1]))
            for teacher, target in [(TEACHERS[0], [0, 4, 3, 2, 1]), (TEACHERS[1], [0, 0, 0, 0, 0])]:
                loss = FixedPseudoFY.__new__(FixedPseudoFY)
                nn.Module.__init__(loss)
                loss.packed, loss.teacher = packed, teacher
                capture = Mock(return_value=torch.tensor([.1]))
                loss.fy = capture
                # Changing reactivity supplied during training must not change targets.
                for profile in (torch.zeros(5), torch.ones(5)*100):
                    loss(['ACGU'], [profile], fname=['rna0'])
                    torch.testing.assert_close(capture.call_args.args[1][0], torch.tensor(target))
                with self.assertRaises(ValueError):
                    loss(['AAAA'], [torch.zeros(5)], fname=['rna0'])
                with self.assertRaises(ValueError):
                    loss(['ACGU'], [torch.zeros(5)])


if __name__ == '__main__':
    unittest.main()
