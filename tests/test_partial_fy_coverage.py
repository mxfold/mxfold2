"""Sampler, diagnostic and exact continuation tests for the new pilot."""
from pathlib import Path
import contextlib
import io
import tempfile
import unittest

import numpy as np
import torch

from scripts.rnagym import partial_fy_coverage as pilot


class MemoryTeachers:
    def __init__(self, model, records):
        self.names = [r[0] for r in records]
        self.sequences = [r[1] for r in records]
        self.indices = {name: i for i, name in enumerate(self.names)}
        self.records, self.labels = records, []
        with pilot.evaluation_mode(model, -1):
            for _, sequence, values in records:
                _, _, _, parameters, _ = model([sequence], return_param=True)
                _, _, guided = model([sequence], param=parameters,
                                      pseudoenergy=[pilot.fixed.pseudoenergy(values['target'], len(sequence))])
                self.labels.append(torch.tensor(guided[0]))

    def __getitem__(self, index):
        return self.records[index]

    def __len__(self):
        return len(self.records)

    def target(self, teacher, index):
        if teacher != 'baseline_assisted_pseudo':
            raise ValueError(teacher)
        return self.labels[index].clone()


class ChemicalCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.model = pilot.native_lstm(pilot.common.load_model(pilot.common.BASELINE, -1))
        sequences = ['GCGAAAACGC', 'GGGAAACCC', 'GUGCAAAUGCAC']
        cls.shapes = []
        cls.structures = []
        for index, sequence in enumerate(sequences):
            profile = torch.full((len(sequence)+1,), .1)
            profile[0], profile[1], profile[2] = -999., 1.3, .9
            cls.shapes.append((f'shape-{index}', sequence, dict(target=profile, type='SHAPE')))
            cls.structures.append((f'structure-{index}', sequence,
                                   dict(target=torch.zeros(len(sequence)+1, dtype=torch.long))))
        cls.packed = MemoryTeachers(cls.model, cls.shapes)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def entry(self, method):
        return next(e for e in pilot.entries() if e['method'] == method)

    def test_each_record_is_seen_exactly_once_per_pass_and_methods_are_matched(self):
        orders = []
        for method in pilot.METHODS:
            engine = pilot.Engine(self.entry(method), self.structures, self.shapes, self.packed, -1)
            sequence = [engine.next_pair()[1] for _ in range(9)]
            for start in (0, 3, 6):
                self.assertEqual(sorted(sequence[start:start+3]), [0, 1, 2])
            self.assertEqual(engine.draws, 9)
            orders.append(sequence)
        self.assertTrue(all(order == orders[0] for order in orders))

    def test_packed_heldout_records_cannot_expand_training_pool(self):
        engine = pilot.Engine(self.entry('fixed_assisted'), self.structures, self.shapes[:2], self.packed, -1)
        self.assertEqual(len(engine.shapes), 2)
        self.assertEqual({engine.next_pair()[1] for _ in range(8)}, {0, 1})

    def test_diagnostic_gaps_match_actual_loss_implementations(self):
        with pilot.evaluation_mode(self.model, -1):
            rows, _ = pilot.diagnostic_rows(self.model, dict(train=self.shapes), self.packed)
            for row, (name, sequence, values) in zip(rows, self.shapes):
                for method, key in [('fixed_assisted', 'fixed_assisted_fy'), ('shape_fy', 'shape_fy'),
                                    ('partial_fy_t08', 'partial_fy_t08'), ('partial_fy_t12', 'partial_fy_t12')]:
                    loss = pilot.auxiliary(self.model, method, self.packed, perturb=0.)
                    expected = float(loss([sequence], [values['target']], fname=[name]).sum())
                    self.assertAlmostEqual(row[key], expected, delta=2e-6)
                self.assertEqual(row['teacher_changed_from_initial'], 0.)

    def test_diagnostics_restore_training_flags_and_rng_and_turnover_is_zero(self):
        model = self.model.train()
        model.zuker.eval()  # Mixed per-module mode must be preserved, not just top-level.
        modes = [m.training for m in model.modules()]
        before = torch.get_rng_state().clone()
        with pilot.evaluation_mode(model, -1):
            rows, state = pilot.diagnostic_rows(model, dict(train=self.shapes), self.packed)
        with pilot.evaluation_mode(model, -1):
            repeated, _ = pilot.diagnostic_rows(model, dict(train=self.shapes), self.packed, state)
        self.assertEqual(modes, [m.training for m in model.modules()])
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        for row in repeated:
            self.assertEqual(row['teacher_changed_from_previous'], 0.)
            self.assertEqual(row['prediction_changed_from_previous'], 0.)

    def test_real_training_is_unchanged_by_diagnostics_and_resume_for_all_methods(self):
        with tempfile.TemporaryDirectory(prefix='coverage-test-') as directory:
            path = Path(directory) / 'checkpoint.pth'
            def run(method, diagnose, split):
                engine = pilot.Engine(self.entry(method), self.structures, self.shapes, self.packed, -1)
                if diagnose or split:
                    engine.advance(2, diagnose=diagnose)
                    if diagnose:
                        with pilot.evaluation_mode(engine.model, -1):
                            pilot.diagnostic_rows(engine.model, dict(train=self.shapes[:1]), self.packed)
                    if split:
                        pilot.shared.save_state(path, engine.snapshot())
                        del engine
                        engine = pilot.Engine(self.entry(method), self.structures, self.shapes, self.packed, -1)
                        engine.restore(path)
                engine.advance(5, diagnose=diagnose)
                state = engine.snapshot()
                state['model_state_dict'] = {k: v.detach().clone() for k, v in engine.model.state_dict().items()}
                return state
            for method in pilot.METHODS:
                with contextlib.redirect_stdout(io.StringIO()):
                    reference = run(method, False, False)
                    pilot.compare_snapshots(reference, run(method, True, False), -1)
                    pilot.compare_snapshots(reference, run(method, True, True), -1)

    def test_selection_respects_budget_and_earliest_tie(self):
        curve = [dict(updates=u, F1=v) for u, v in [(0, .6), (100, .6), (3000, .61), (6000, .63), (9000, .62)]]
        self.assertEqual(pilot.shared.choose(curve[:2])['updates'], 0)
        self.assertEqual(pilot.shared.choose([r for r in curve if r['updates'] <= 3000])['updates'], 3000)
        self.assertEqual(pilot.shared.choose(curve)['updates'], 6000)


if __name__ == '__main__':
    unittest.main()
