import importlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from scripts.rnagym.evaluate_rnagym import chemical_metrics
from scripts.rnagym import replay20_full_test as workflow


class Replay20Test(unittest.TestCase):
    def test_submission_dependencies_and_parallel_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = {}
            for m in workflow.REFERENCES:
                for fraction in (10, 20):
                    models[f'{m}_{fraction}'] = {'method': m, 'replay_percent': fraction}
            models['baseline'] = {'method': 'baseline', 'replay_percent': 0}
            workflow.write_json(root / 'comparison_plan.json', {
                'models': models, 'shards': 16, 'parallel_tasks_per_model': 4})
            with patch.object(workflow.subprocess, 'check_output', side_effect=[str(i) for i in range(100, 129)]) as call:
                workflow.submit(root)
            jobs = [json.loads(line) for line in (root / 'submissions.jsonl').read_text().splitlines()]
            self.assertEqual(len(jobs), 29)
            ends = {j['method']: j['job_id'] for j in jobs if j['action'] == 'train' and j['epoch'] == 3}
            for job in jobs:
                if job['action'] == 'predict':
                    spec = models[job['model']]
                    if spec['replay_percent'] == 20:
                        self.assertIn(ends[spec['method']], job['hold'])
                    self.assertEqual(job['hold'][:2], ['100', '101'])
            self.assertEqual(len(jobs[-1]['hold']), 7)
            arrays = [c.args[0] for c in call.call_args_list if '-t' in c.args[0]]
            self.assertEqual(len(arrays), 7)
            self.assertTrue(all(c[c.index('-t') + 1] == '1-16' and c[c.index('-tc') + 1] == '4' for c in arrays))

    def test_all_methods_cover_chemical_once_and_cycle_replay(self):
        indices = np.arange(112815)
        expected = None
        for method, runner in workflow.RUNNERS.items():
            module = importlib.import_module('scripts.rnagym.' + runner.removesuffix('.py'))
            for fraction, count in ((.1, 12535), (.2, 28204)):
                self.assertEqual(math.ceil(len(indices) * fraction / (1 - fraction)), count)
                slots = module.epoch_slots(indices, 3166, 1, count, 0)
                chem = sorted(i for kind, i in slots if kind == 'chemical')
                self.assertEqual(chem, indices.tolist())
                self.assertEqual(sum(kind == 'replay' for kind, _ in slots), count)
                self.assertEqual(module.epoch_slots(indices, 3166, 1, count, 0), slots)
                if fraction == .2:
                    if expected is None:
                        expected = slots
                    self.assertEqual(expected, slots)
                combined = module.make_replay_stream(3166, 0, 3 * count, 0)
                self.assertLessEqual(np.ptp(np.bincount(combined, minlength=3166)), 1)
                self.assertEqual(combined[count:2 * count], module.make_replay_stream(3166, count, count, 0))

    def test_completed_epoch_checkpoint_survives_later_updates(self):
        for method, runner in workflow.RUNNERS.items():
            module = importlib.import_module('scripts.rnagym.' + runner.removesuffix('.py'))
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'runs').mkdir()
                model = torch.nn.Linear(2, 1)
                optimizer = torch.optim.SGD(model.parameters(), lr=.1)
                shapes = torch.nn.Linear(1, 1)
                state = {'completed_epoch': 1, 'active_epoch': 0}
                args = [root, model, optimizer] + ([shapes] if method == 'nll' else [])
                module.save_checkpoint(*args, state)
                cp = root / 'runs/epoch-001.pt'
                old_hash = workflow.digest(cp)
                with torch.no_grad():
                    model.weight.add_(1)
                state = {'completed_epoch': 1, 'active_epoch': 2}
                module.save_checkpoint(*args, state)
                self.assertEqual(workflow.digest(cp), old_hash)
                self.assertNotEqual(workflow.digest(root / 'runs/latest.pt'), old_hash)

    def test_global_metrics_match_pinned_official_code(self):
        from scripts.rnagym.official_metrics_reference.utils import compute_performance_metrics
        rng = np.random.default_rng(17)
        for ties in (False, True):
            frame = []
            true_parts, pred_parts = [], []
            for length in (7, 51, 1001):
                true, pred = rng.normal(.5, .7, (2, length))
                if ties:
                    true, pred = np.round(true, 1), np.round(pred, 1)
                true[0] = np.nan
                frame.append({'reactivity_DMS': '[' + ','.join(map(str, true)) + ']',
                              'reactivity_2A3': '[' + ','.join(map(str, true)) + ']',
                              'prediction_MXfold2': pred})
                true_parts.append(true)
                pred_parts.append(pred)
            official = compute_performance_metrics(pd.DataFrame(frame), 'MXfold2')
            result = chemical_metrics(np.concatenate(true_parts), np.concatenate(pred_parts))
            for modifier in official:
                for ours, theirs in (('auc', 'AUC'), ('mcc', 'MCC'), ('f1', 'F1-Score'), ('mae', 'MAE')):
                    self.assertAlmostEqual(result[ours], modifier[theirs], places=13)

    def test_shards_join_before_medians_and_require_complete_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / 'evaluation/demo'
            folder.mkdir(parents=True)
            source = str(Path(__file__).resolve().parents[1])
            workflow.write_json(root / 'comparison_plan.json', {
                'shards': 2, 'models': {'demo': {'source': source}}})
            metadata = {'unique_sequences': 2, 'shards': {}}
            truth = [np.array([.1, .2, .3]), np.array([.8, .9])]
            pred = [np.array([.9, .8, .7]), np.array([.2, .1])]
            for shard in (1, 2):
                metadata['shards'][str(shard)] = {'unique_sequences': 1, 'modifiers': {m: 1 for m in workflow.MODIFIERS}}
                workflow.write_json(folder / f'shard-{shard}.json', {
                    'success': True, 'unique_sequences': 1, 'checkpoint_sha256': 'same'})
                values = {}
                for m in workflow.MODIFIERS:
                    values.update({f'{m}_true': truth[shard - 1], f'{m}_pred': pred[shard - 1],
                                   f'{m}_n_sequences': np.array([1])})
                np.savez(folder / f'shard-{shard}.npz', **values)
            workflow.write_json(root / 'test_data/complete.json', metadata)
            workflow.write_json(folder / 'structure.json', {})
            workflow.aggregate(root, 'demo')
            result = json.loads((folder / 'metrics.json').read_text())
            expected = chemical_metrics(np.concatenate(truth), np.concatenate(pred))
            self.assertEqual(result['chemical']['DMS']['auc'], expected['auc'])
            self.assertEqual(result['chemical']['DMS']['f1'], expected['f1'])
            (folder / 'shard-2.json').unlink()
            with self.assertRaises(FileNotFoundError):
                workflow.aggregate(root, 'demo')


if __name__ == '__main__':
    unittest.main()
