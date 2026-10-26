import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from mxfold2.train import Train
from scripts.rnagym import rank_improvement_experiments as experiments


def parse_command(command):
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    Train.add_args(parser.add_subparsers())
    return parser.parse_args(command[3:])


def selected_curve(f1, epoch=2):
    curve = [{'epoch': index, 'TrainSetB': {'mean_F1': f1 if index == epoch else f1 - .01}}
             for index in experiments.CANDIDATE_EPOCHS]
    return {'epoch': epoch, 'curve': curve}


class RankImprovementExperimentsTest(unittest.TestCase):
    def test_all_primary_commands_parse_and_match_declared_budgets(self):
        entries = experiments.primary_entries()
        self.assertEqual(len(entries), 21)
        self.assertEqual(len({entry['condition'] for entry in entries}), 7)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {'initial_checkpoint': str(experiments.revision.BASE_CP)}
            for entry in entries:
                with self.subTest(condition=entry['condition'], seed=entry['seed']):
                    args = parse_command(experiments.train_command(root, plan, entry, 'primary', 1))
                    self.assertEqual(args.samples_per_epoch, entry['draws'])
                    self.assertEqual(args.grad_accum_steps, entry['accumulation'])
                    self.assertEqual(args.shape_loss_weight, entry['shape_weight'])
                    self.assertEqual(args.perturb, .1)
                    self.assertEqual(args.score_loss_weight, 0)
                    self.assertEqual(args.optimizer, 'AdamW')
                    self.assertEqual(args.lr, 1e-4)
                    self.assertEqual(args.l2_weight, 1e-4)
                    self.assertEqual(args.init_param, plan['initial_checkpoint'])
                    if entry['method'] == 'structure':
                        self.assertEqual(args.bpseq_loss_weight, 1)
                        self.assertEqual(args.task, 'Folding')
                    else:
                        self.assertEqual(args.shape_loss_func, entry['method'])
                        self.assertEqual(args.replay_fraction, entry['replay'])

    def test_chemistry_null_control_preserves_sampler_and_structural_coefficients(self):
        entries = experiments.primary_entries()
        rank = next(entry for entry in entries if entry['condition'] == 'rank_delta002')
        null = next(entry for entry in entries if entry['condition'] == 'mixed50_no_chemical_loss')
        plan = {'initial_checkpoint': str(experiments.revision.BASE_CP)}
        rank_args = parse_command(experiments.train_command(Path('/tmp/experiment'), plan, rank, 'primary', 1))
        null_args = parse_command(experiments.train_command(Path('/tmp/experiment'), plan, null, 'primary', 1))
        for field in ['seed', 'samples_per_epoch', 'grad_accum_steps', 'replay_fraction',
                      'shape', 'input', 'input_2', 'second_loss_weight', 'bpseq_loss_weight',
                      'shape_loss_func', 'perturb', 'shape_perturb', 'shape_rank_delta']:
            self.assertEqual(getattr(rank_args, field), getattr(null_args, field), field)
        self.assertEqual(null_args.shape_loss_weight, 0)
        self.assertEqual(rank_args.shape_loss_weight, .5)

    def test_original_rank_winner_parser_equivalence(self):
        source = experiments.SOURCE / 'budget/imle_rank_shape_replay50_seed1234/segment-1/command.json'
        if not source.is_file():
            self.skipTest('Existing revision-v2 source command is unavailable')
        original = parse_command(json.loads(source.read_text()))
        entry = experiments.primary_entries()[0]
        new = parse_command(experiments.train_command(Path('/tmp/experiment'),
              {'initial_checkpoint': str(experiments.revision.BASE_CP)}, entry, 'primary', 1))
        for field in ['seed', 'samples_per_epoch', 'grad_accum_steps', 'replay_fraction',
                      'optimizer', 'lr', 'l2_weight', 'score_loss_weight', 'clip_grad_norm',
                      'perturb', 'shape_perturb', 'loss_func', 'shape_loss_func', 'task',
                      'bpseq_loss_weight', 'second_loss_weight', 'shape_loss_weight',
                      'shape_nu', 'shape_model', 'shape_margin', 'shape_rank_tau',
                      'shape_rank_weighting', 'shape_rank_delta', 'shape_rank_targets',
                      'shape_rank_teacher_temperature', 'shape_rank_teacher_normalization',
                      'shape_rank_sample_aggregation', 'shape_pseudo_fy_weight',
                      'shape_imle_alignment', 'shape_imle_normalization',
                      'shape_imle_estimator_scale', 'shape_imle_difference', 'shape_imle_samples',
                      'weight_schedule']:
            self.assertEqual(getattr(original, field), getattr(new, field), field)

    def test_resume_keeps_experiment_checkpoint(self):
        entry = experiments.primary_entries()[0]
        plan = {'initial_checkpoint': '/some/original/epoch-100'}
        root = Path('/tmp/experiment')
        args = parse_command(experiments.train_command(root, plan, entry, 'primary', 3))
        self.assertFalse(args.init_param)
        self.assertEqual(args.resume, str(experiments.checkpoint_path(root, 'primary', entry, 2)))
        self.assertEqual(args.epochs, 3)

    def test_earliest_epoch_wins_validation_ties(self):
        curve = [{'epoch': epoch, 'TrainSetB': {'mean_F1': .7}} for epoch in [5, 2, 1, 0]]
        self.assertEqual(experiments.choose_epoch(curve)['epoch'], 0)

    def test_stage2_uses_validation_only_and_stable_tie_order(self):
        entries = experiments.primary_entries()
        selections = {}
        for entry in entries:
            # Test values favor delta .1 but are deliberately unrelated.
            selections[entry['name']] = {**selected_curve(.72 if entry.get('delta') == .05 else .70),
                'TestSetB': {'mean_F1': .999 if entry.get('delta') == .10 else .1}}
        delta, scores = experiments.choose_stage2_delta(entries, selections)
        self.assertEqual(delta, .05)
        self.assertEqual(len(scores), 3)
        for entry in entries:
            selections[entry['name']] = selected_curve(.71)
        self.assertEqual(experiments.choose_stage2_delta(entries, selections)[0], .002)

    def test_missing_dependency_or_changed_plan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'plan.json').write_text('{}\n')
            with self.assertRaises(FileNotFoundError):
                experiments.check_marker(root, root / 'dependency.json')
            experiments.write_marker(root, root / 'dependency.json', epoch=1)
            self.assertTrue(experiments.check_marker(root, root / 'dependency.json')['success'])
            (root / 'plan.json').write_text('{"changed":true}\n')
            with self.assertRaises(ValueError):
                experiments.check_marker(root, root / 'dependency.json')

    def test_stage2_refuses_unfinalized_delta(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'plan.json').write_text('{}\n')
            plan = {'stage2': experiments.stage2_entries()}
            with self.assertRaises(FileNotFoundError):
                experiments.entry_for(root, plan, 'stage2', 0)
            experiments.write_marker(root, root / 'stage2_plan.json', delta=.05)
            entry = experiments.entry_for(root, plan, 'stage2', 0)
            self.assertEqual(entry['delta'], .05)
            self.assertIsNone(plan['stage2'][0]['delta'])

    def test_bootstrap_uses_paired_differences(self):
        interval = experiments.paired_bootstrap(np.repeat(.015, 430), samples=500, seed=1)
        self.assertAlmostEqual(interval['difference'], .015)
        self.assertAlmostEqual(interval['ci95_low'], .015)
        self.assertAlmostEqual(interval['ci95_high'], .015)
        with self.assertRaises(ValueError):
            experiments.paired_bootstrap([.1, float('nan')])

    def test_test_ids_align_and_missing_ids_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'results.csv'
            pd.DataFrame({'name': ['b', 'a'], 'F': [.7, .4]}).to_csv(path, index=False)
            aligned = experiments.aligned_f1(path, ['a', 'b'])
            np.testing.assert_allclose(aligned.to_numpy(), [.4, .7])
            with self.assertRaises(ValueError):
                experiments.aligned_f1(path, ['a', 'missing'])

    def test_summary_uses_all_430_ids_and_reports_micro_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'plan.json').write_text('{}\n')
            baseline_path = root / 'baseline/TestSetB/result.csv'
            baseline_path.parent.mkdir(parents=True)
            identifiers = [f'rna-{index}' for index in range(430)]
            pd.DataFrame({'name': identifiers, 'F': np.repeat(.6, 430)}).to_csv(baseline_path, index=False)
            experiments.write_marker(root, root / 'reference.json')
            entries = experiments.primary_entries()
            chemical = {'All': {'auc': .7, 'mae': .2, 'mcc': .4, 'f1': .7}}
            for entry in entries:
                folder = root / 'primary' / entry['name']
                (folder / 'TestSetB').mkdir(parents=True)
                selection = selected_curve(.72)
                for record in selection['curve']:
                    record['RNAgym_train_validation'] = chemical
                (folder / 'selection.json').write_text(json.dumps(selection))
                pd.DataFrame({'name': identifiers[::-1], 'F': np.repeat(.62, 430)}).to_csv(
                    folder / 'TestSetB/result.csv', index=False)
                experiments.write_marker(root, folder / 'selected_completed.json', TestSetB={
                    'mean_SEN': .65, 'mean_PPV': .61, 'micro_F1': .60, 'micro_SEN': .63, 'micro_PPV': .58})
            plan = {'primary': entries, 'summary_protocol': {'expected_test_ids': 430,
                    'bootstrap_samples': 100, 'bootstrap_seed': 1},
                    'seed_scope': 'fixed baseline', 'comparison_limits': 'fixed baseline'}
            experiments.summary_stage(root, plan, primary_only=True)
            summary = pd.read_csv(root / 'primary_seed_summary.csv')
            self.assertEqual(len(summary), 7)
            np.testing.assert_allclose(summary.difference, .02)
            np.testing.assert_allclose(summary.TestSetB_micro_F1_mean, .60)
            self.assertTrue(summary.meets_predeclared_target.all())
            self.assertEqual(len(pd.read_csv(root / 'primary_per_sequence.csv')), 21 * 430)
            self.assertTrue(experiments.check_marker(root, root / 'primary_completed.json')['success'])

    def test_submission_dag_and_resume_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'plan.json').write_text('{}\n')
            plan = {'primary': experiments.primary_entries(), 'stage2': experiments.stage2_entries(),
                    'scheduler': {'concurrency': 2}}
            commands = []
            def fake_submit(command, **kwargs):
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, f'{900 + len(commands)}.1-21:1\n', '')
            with patch.object(experiments.subprocess, 'run', side_effect=fake_submit):
                experiments.locked_submit(root, plan)
                self.assertEqual(len(commands), 6)
                run_commands = [command for command in commands if any('IMPROVEMENT_ACTION=run,' in token for token in command)]
                self.assertEqual(len(run_commands), 2)
                self.assertEqual(run_commands[0][run_commands[0].index('-t') + 1], '1-21')
                self.assertEqual(run_commands[1][run_commands[1].index('-t') + 1], '1-3')
                self.assertIn('h_rt=01:00:00', run_commands[0])
                for command in commands[1:]:
                    self.assertIn('-hold_jid', command)
                experiments.locked_submit(root, plan, resume=True)
                self.assertEqual(len(commands), 6)
                with self.assertRaises(FileExistsError):
                    experiments.locked_submit(root, plan)


if __name__ == '__main__':
    unittest.main()
