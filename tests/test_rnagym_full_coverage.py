import json
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from mxfold2.rnagym import coverage_order, measurement_mask, observation_confidence
from mxfold2.loss.chemical_probe_loss import ChemicalProbeLoss
from scripts.rnagym.full_coverage_experiments import (
    checkpoint, choose_cycle, learning_rate, official_clusters, restore, UnionFind,
    train_chunk, probe_correlation, worker_entry,
)


def observation(sequence, modifier, values, confidence=None, condition=0):
    values = np.array(values, np.float32)
    return {'modifier': modifier, 'values': torch.from_numpy(values),
            'mask': torch.from_numpy(measurement_mask(sequence, modifier, values)),
            'confidence': torch.tensor(confidence if confidence is not None else [1.] * len(values)),
            'condition_id': condition}


class CoverageTests(unittest.TestCase):
    def test_resume_cursor_reconstructs_without_repeating_or_omitting(self):
        indices = np.array([1, 4, 7, 8, 20, 21, 100])
        first = coverage_order(indices, 1234, 0)
        resumed = coverage_order(indices, 1234, 0)[3:]
        np.testing.assert_array_equal(np.concatenate([first[:3], resumed]), first)
        self.assertEqual(set(first), set(indices))
        self.assertEqual(len(first), len(set(first)))
        self.assertFalse(np.array_equal(first, coverage_order(indices, 1234, 1)))

    def test_order_does_not_change_torch_or_numpy_training_rng(self):
        np.random.seed(3)
        before = np.random.get_state()
        coverage_order(np.arange(8), 1234, 0)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])

    def test_duplicate_indices_rejected(self):
        with self.assertRaises(ValueError):
            coverage_order([1, 1], 1, 0)

    def test_probe_masks_keep_negative_background_values(self):
        np.testing.assert_array_equal(measurement_mask('ACGUAC', 'DMS',
            np.array([-.2, .3, 10., 10., np.nan, -999.])), [True, True, False, False, False, False])
        np.testing.assert_array_equal(measurement_mask('ACGU', 'CMCT', np.ones(4)), [False, False, True, True])

    def test_copied_error_does_not_create_inverse_reactivity_weights(self):
        values = np.array([.01, .1, 10.], np.float32)
        confidence = observation_confidence(values, values, 2., copied_error=True)
        np.testing.assert_allclose(confidence, np.full(3, 2 / 3))

    def test_missing_and_negative_errors_fall_back_to_snr(self):
        confidence = observation_confidence([.1, .3, .9], [np.nan, -.2, 100.], 1., copied_error=False)
        np.testing.assert_allclose(confidence[:2], [.5, .5])
        self.assertGreater(confidence[2], 0)
        self.assertLess(confidence[2], confidence[0])

    def test_union_same_sequence_clusters_transitively(self):
        u = UnionFind()
        u.union('b', 'c')
        u.union('a', 'b')
        self.assertEqual(u.find('c'), u.find('a'))

    def test_official_parquet_header_record_is_recovered(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'clusters.parquet'
            pd.DataFrame({'rep+alias': ['rep+alias'], 'rep+alias.1': ['member']}).to_parquet(path)
            mapping = official_clusters(path)
            self.assertEqual(mapping['rep'], 'rep+alias')
            self.assertEqual(mapping['alias'], 'rep+alias')
            self.assertEqual(mapping['member'], 'rep+alias')


class ChemicalLossTests(unittest.TestCase):
    def setUp(self):
        self.loss = ChemicalProbeLoss(torch.nn.Linear(1, 1))

    def test_dms_gu_bases_receive_no_chemical_gradient(self):
        sequence = 'AAGUCC'
        obs = observation(sequence, 'DMS', [.1, 1., 100., 100., .2, .9])
        paired = torch.zeros(6, requires_grad=True)
        value = self.loss.rank_objective(sequence, paired, [obs])
        value.backward()
        self.assertLess(float(paired.grad[0]), 0)
        self.assertGreater(float(paired.grad[1]), 0)
        torch.testing.assert_close(paired.grad[2:4], torch.zeros(2))

    def test_dms_does_not_compare_a_against_c_background(self):
        sequence = 'AACCGU'
        obs = observation(sequence, 'DMS', [.1, .1, 1., 1., 10., 10.])
        paired = torch.zeros(6, requires_grad=True)
        value = self.loss.rank_objective(sequence, paired, [obs])
        value.backward()
        torch.testing.assert_close(paired.grad, torch.zeros(6))

    def test_duplicate_replicate_cannot_multiply_sequence_weight(self):
        sequence = 'ACGU'
        obs = observation(sequence, '2A3', [.1, .2, .8, 1.])
        p = torch.tensor([1., 1., 0., 0.], requires_grad=True)
        torch.testing.assert_close(self.loss.rank_objective(sequence, p, [obs]),
                                   self.loss.rank_objective(sequence, p, [obs, obs]))

    def test_lower_confidence_reduces_gradient_magnitude(self):
        sequence = 'ACGU'
        p1, p2 = torch.zeros(4, requires_grad=True), torch.zeros(4, requires_grad=True)
        self.loss.rank_objective(sequence, p1, [observation(sequence, '2A3', [.1, .2, .8, 1.])]).backward()
        self.loss.rank_objective(sequence, p2, [observation(sequence, '2A3', [.1, .2, .8, 1.], [.5] * 4)]).backward()
        torch.testing.assert_close(p2.grad, p1.grad * .25)

    def test_soft_fy_uses_only_positive_evidence(self):
        sequence = 'ACGU'
        obs = observation(sequence, '2A3', [-.2, 0., .5, 100.])
        energy = self.loss.chemical_energy(sequence, [obs], 'cpu')
        torch.testing.assert_close(energy[:3], torch.zeros(3))
        self.assertLessEqual(float(energy[3]), .5)
        self.assertGreater(float(energy[3]), 0)

    def test_missing_measurements_do_not_dilute_observed_signal(self):
        sequence = 'ACGU'
        first = observation(sequence, '2A3', [1., 1., 1., 1.])
        missing = observation(sequence, 'DMS', [np.nan] * 4)
        torch.testing.assert_close(self.loss.chemical_energy(sequence, [first], 'cpu'),
                                   self.loss.chemical_energy(sequence, [first, missing], 'cpu'))

    def test_different_conditions_cannot_be_averaged_into_one_fy_teacher(self):
        with self.assertRaises(ValueError):
            self.loss.chemical_energy('AC', [observation('AC', '2A3', [1., 0.], condition=0),
                                              observation('AC', '2A3', [0., 1.], condition=1)], 'cpu')

    def test_no_valid_measurements_have_zero_loss_and_gradient(self):
        p = torch.zeros(4, requires_grad=True)
        value = self.loss.rank_objective('ACGU', p, [observation('ACGU', '2A3', [np.nan] * 4)])
        value.backward()
        self.assertEqual(float(value.detach()), 0)
        torch.testing.assert_close(p.grad, torch.zeros(4))


class ProtocolTests(unittest.TestCase):
    def test_scheduler_wrapper_handles_nonarray_undefined_and_array_index(self):
        script = Path(__file__).resolve().parents[1] / 'scripts/rnagym/qsub_full_coverage_experiments.sh'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / '.venv/bin/python'
            executable.parent.mkdir(parents=True)
            executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            executable.chmod(0o755)
            for value, expected in [('undefined', '1'), ('7', '7')]:
                env = {**os.environ, 'PROJECT_ROOT': str(root), 'FULL_ROOT': str(root),
                       'FULL_ACTION': 'preflight', 'SGE_TASK_ID': value}
                args = subprocess.run(['bash', str(script)], env=env, check=True,
                                      capture_output=True, text=True).stdout.splitlines()
                self.assertEqual(args[args.index('--index') + 1], expected)

    def test_dms_validation_ignores_gu_and_cross_base_background(self):
        self.assertAlmostEqual(probe_correlation('AACCGU', 'DMS',
            np.array([.1, .8, 1., 2., 0., 0.]), np.array([.1, .8, .1, .8, 1., 1.])), 1.)

    def test_worker_index_cannot_silently_wrap_to_last_arm(self):
        with self.assertRaises(ValueError):
            worker_entry({'development_seed': 1234}, 'development', 0)

    def test_cycle_boundary_and_idempotent_chunks_cover_every_sequence(self):
        # Seven sequence groups expose the non-multiple-of-four boundary that
        # could otherwise skip/repeat members across jobs and coverage cycles.
        from scripts.rnagym import full_coverage_experiments as workflow
        calls = []
        class FakeChemical:
            def __init__(self, model, **kwargs):
                self.model = model
                self.last_imle_stats = {'flip_fraction': [1.]}
            def __call__(self, seq, observations):
                calls.append(seq)
                return self.model(torch.ones(1, 1)).square().mean()
        class FakeStructural:
            def __init__(self, model, **kwargs):
                self.model = model
            def __call__(self, *args):
                return self.model(torch.ones(1, 1)).square().mean()
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            data = out / 'data'
            data.mkdir()
            sequences = ['A' * (i + 2) + 'C' for i in range(7)]
            with (data / 'sequences.tsv').open('w') as f:
                f.write('index\tsequence\tcluster\tsplit\n')
                for i, s in enumerate(sequences):
                    f.write(f'{i}\t{s}\tc{i}\ttrain\n')
            offsets = np.concatenate([[0], np.cumsum(list(map(len, sequences)))])
            np.save(data / 'values.npy', np.ones(offsets[-1], np.float32))
            np.save(data / 'confidence.npy', np.ones(offsets[-1], np.float32))
            np.save(data / 'observation_offsets.npy', offsets)
            np.save(data / 'sequence_offsets.npy', np.arange(8))
            np.save(data / 'modifiers.npy', np.zeros(7, np.int8))
            np.save(data / 'conditions.npy', np.zeros(7, np.int32))
            (data / 'manifest.json').write_text('{}')
            bpseq = out / 'a.bpseq'
            bpseq.write_text('1 A 0\n2 C 0\n')
            (out / 'TrainSetA.lst').write_text(str(bpseq) + '\n')
            plan = {'initial_checkpoint': 'dummy', 'development_cycles': 2, 'sequence_chunk': 4,
                    'chemical_weight': .5, 'fy_weight': .25, 'lr': .001, 'min_lr': .0001,
                    'warmup_updates': 1, 'weight_decay': 0, 'clip_grad_norm': 1}
            (out / 'plan.json').write_text(json.dumps(plan))
            with patch.object(workflow, 'verify_plan', return_value=plan), \
                 patch.object(workflow.revision, 'load_model', side_effect=lambda *_args, **_kwargs: torch.nn.Linear(1, 1)), \
                 patch.object(workflow, 'ChemicalProbeLoss', FakeChemical), \
                 patch.object(workflow, 'FenchelYoungLoss', FakeStructural), \
                 patch.object(workflow, 'validation', return_value={}):
                for chunk in range(1, 6):
                    train_chunk(out, condition='probe_rank', seed=1234, phase='development', chunk=chunk, gpu=-1)
                train_chunk(out, condition='probe_rank', seed=1234, phase='development', chunk=5, gpu=-1)
            self.assertEqual(len(calls), 14)
            self.assertEqual(set(calls[:7]), set(sequences))
            self.assertEqual(set(calls[7:]), set(sequences))
            state = json.loads((out / 'development/probe_rank_seed1234/chunk-005.json').read_text())
            self.assertEqual(state['cycle'], 2)
            self.assertEqual(state['cursor'], 0)
            self.assertEqual(state['chemical_observations'], 14)
            self.assertEqual(state['sequence_slots'], 14)
            self.assertTrue(state['already_complete'])

    def test_cycle_zero_or_partial_coverage_is_never_selected(self):
        with self.assertRaises(ValueError):
            choose_cycle([{'cycle': 0, 'coverage_fraction': 1.}, {'cycle': 1, 'coverage_fraction': .99}], {}, ood_sequences=100)

    def test_structural_ood_selection_cannot_read_test_scores(self):
        curves = [{'cycle': 1, 'coverage_fraction': 1., 'metrics': {'TrainSetB_ood': {'cluster_macro_F1': .7}, 'TestSetB': {'F1': 0}}},
                  {'cycle': 2, 'coverage_fraction': 1., 'metrics': {'TrainSetB_ood': {'cluster_macro_F1': .6}, 'TestSetB': {'F1': 1}}}]
        self.assertEqual(choose_cycle(curves, {}, ood_sequences=100)['cycle'], 1)

    def test_checkpoint_restores_optimizer_cursor_and_all_cpu_rng(self):
        with tempfile.TemporaryDirectory() as temp:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters())
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            state = {'cycle': 0, 'cursor': 17, 'updates': 4}
            checkpoint(Path(temp), model, optimizer, state)
            expected = (torch.rand(2), random.random(), np.random.random())
            fresh = torch.nn.Linear(2, 1)
            fresh_optimizer = torch.optim.AdamW(fresh.parameters())
            result = restore(Path(temp) / 'latest.pt', fresh, fresh_optimizer)
            actual = (torch.rand(2), random.random(), np.random.random())
            torch.testing.assert_close(expected[0], actual[0])
            self.assertEqual(expected[1:], actual[1:])
            self.assertEqual(result, state)
            self.assertEqual(len(fresh_optimizer.state), len(optimizer.state))
            for left, right in zip(model.parameters(), fresh.parameters()):
                torch.testing.assert_close(left, right)

    def test_warmup_and_decay_have_fixed_full_cycle_length(self):
        plan = {'lr': 3e-5, 'min_lr': 3e-6, 'warmup_updates': 100}
        self.assertLess(learning_rate(plan, 0, 1000), learning_rate(plan, 99, 1000))
        self.assertAlmostEqual(learning_rate(plan, 1000, 1000), 3e-6)


if __name__ == '__main__':
    unittest.main()
