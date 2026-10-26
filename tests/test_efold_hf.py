import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.efold_hf.workflow import (
    backward_step, build, clean_observations, pair_metrics, project_pairs,
    dispatch_completed, restore, save_checkpoint, schedule, seed_all,
    initialize_finetuning, optimizer_step, train_worker, training_settings, unpaired_probabilities,
)


class EfoldProtocolTests(unittest.TestCase):
    def test_restored_arms_build_with_matched_nll_settings(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'baseline.conf').write_text('')
            for arm in ('mw_rank_restored', 'nll_restored', 'shape_fy_restored'):
                with patch('mxfold2.train.Train.build_model', return_value=(torch.nn.Linear(1, 1), None)):
                    _, optimizer, _, chemical, shapes, _ = build(root, arm, -1)
                self.assertEqual(training_settings(root, arm)['normalization'], 'rms')
                self.assertEqual(optimizer.param_groups[0]['lr'], .0001)
                self.assertEqual(optimizer.param_groups[0]['weight_decay'], .0001)
                self.assertEqual(training_settings(root, arm)['accumulation'], 4)
                if arm == 'mw_rank_restored':
                    self.assertEqual(type(chemical).__name__, 'FullRNAGymRankLoss')
                    self.assertEqual(chemical.thermo_weight, .125)
                elif arm == 'nll_restored':
                    self.assertEqual(len(shapes), 6)
                    self.assertEqual(chemical.thermo_weight, .125)
                else:
                    self.assertEqual(chemical.base.sl_weight, .125)
                    self.assertEqual(chemical.base.perturb, .1)
                    self.assertEqual(chemical.base.shape_slope, 2.6)
                    self.assertEqual(chemical.base.shape_intercept, -.8)

    def test_pretraining_moments_restore_and_new_Wu_starts_fresh(self):
        from pytorch_optimizer import AdaBelief
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'plan.json').write_text(json.dumps({'pretrain_epochs': 10}))
            model = torch.nn.Linear(1, 1)
            optimizer = AdaBelief(model.parameters(), lr=.001, weight_decay=.01)
            for _ in range(3):
                backward_step(model(torch.ones(1)).square().sum(), model, optimizer)
            save_checkpoint(root/'runs/pretrain/epoch-010.pt', model, optimizer, None, {'epoch': 10})
            target = torch.nn.Linear(1, 1)
            shape = torch.nn.Linear(1, 1)
            restored = AdaBelief([{'params':target.parameters(), 'lr':.0001,'weight_decay':.0001},
                                  {'params':shape.parameters(),'lr':.0001,'weight_decay':.0001}])
            initialize_finetuning(root, 'nll_restored', target, restored, [shape])
            self.assertEqual(restored.param_groups[0]['step'], 3)
            self.assertEqual(restored.param_groups[0]['lr'], .0001)
            for source_parameter, target_parameter in zip(model.parameters(), target.parameters()):
                torch.testing.assert_close(source_parameter, target_parameter, rtol=0, atol=0)
                for key in ('exp_avg', 'exp_avg_var'):
                    torch.testing.assert_close(optimizer.state[source_parameter][key],
                                               restored.state[target_parameter][key], rtol=0, atol=0)
            self.assertEqual(restored.param_groups[1]['step'], 0)
            for parameter in shape.parameters():
                self.assertEqual(int(restored.state[parameter]['exp_avg'].count_nonzero()), 0)
            backward_step(target(torch.ones(1)).square().sum()+shape(torch.ones(1)).square().sum(), target, restored)
            self.assertEqual(restored.param_groups[0]['step'], 4)
            self.assertEqual(restored.param_groups[1]['step'], 1)

    def test_restored_worker_accumulates_and_weights_partial_last_batch(self):
        from pytorch_optimizer import AdaBelief
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan={'pretrain_epochs':10,'finetune_epochs':1,'worker_hours':20}
            (root/'plan.json').write_text(json.dumps(plan))
            (root/'preflight.json').write_text('{"success":true}')
            model=torch.nn.Linear(1,1)
            expected=torch.nn.Linear(1,1)
            expected.load_state_dict(model.state_dict())
            optimizer=AdaBelief(model.parameters(),lr=.0001)
            expected_optimizer=AdaBelief(expected.parameters(),lr=.0001)
            primary=list(range(5)); replay=list(range(10,15))
            def sample_loss(item, *_):
                return (model(torch.ones(1))-item/20).square().mean()
            settings=training_settings(root,'nll_restored')
            order=schedule(5,5,1,seed=settings['seed'],replay_mode=settings['replay_mode'])
            for start in range(0,len(order),4):
                batch=order[start:start+4]
                for kind,index in batch:
                    item=(replay if kind=='replay' else primary)[index]
                    loss=(expected(torch.ones(1))-item/20).square().mean()
                    (loss*(.5 if kind=='primary' else 1.)/len(batch)).backward()
                optimizer_step(expected_optimizer)
            with patch('scripts.efold_hf.workflow.verify',return_value=plan), \
                 patch('scripts.efold_hf.workflow.build',return_value=(model,optimizer,None,object(),None,'cpu')), \
                 patch('scripts.efold_hf.workflow.initialize_finetuning'), \
                 patch('scripts.efold_hf.workflow.read_jsonl',side_effect=lambda p:replay if p.name=='bprna.jsonl' else primary), \
                 patch('scripts.efold_hf.workflow.sequence_loss',side_effect=sample_loss) as loss_call, \
                 patch('scripts.efold_hf.workflow.dispatch_completed'):
                train_worker(root,'nll_restored',-1)
            self.assertEqual(loss_call.call_count,6)
            final=torch.load(root/'runs/nll_restored/epoch-001.pt',weights_only=False)
            self.assertEqual(final['state']['updates'],2)
            self.assertEqual(final['state']['cursor'],6)
            self.assertTrue((root/'runs/nll_restored/initial.pt').exists())
            for a,b in zip(model.parameters(),expected.parameters()):
                torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_restored_replay_cycles_cover_pool_before_repeating(self):
        from scripts.efold_hf.matched_losses import make_replay_stream
        first=make_replay_stream(7,0,7,0)
        second=make_replay_stream(7,7,7,0)
        self.assertEqual(sorted(first),list(range(7)))
        self.assertEqual(sorted(second),list(range(7)))
        self.assertEqual(first+second,make_replay_stream(7,0,14,0))

    def test_mw_arm_builds_raw_auroc_and_keeps_logistic_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'baseline.conf').write_text('')
            for arm, objective in [('mw_rank', 'mann_whitney'), ('rank', 'pairwise_logistic')]:
                with patch('mxfold2.train.Train.build_model',
                           return_value=(torch.nn.Linear(1, 1), None)):
                    _, optimizer, _, chemical, _, _ = build(root, arm, -1)
                self.assertEqual(chemical.objective, objective)
                self.assertEqual(chemical.normalization, 'rms')
                self.assertEqual(chemical.sl_weight, .125)
                self.assertEqual(optimizer.param_groups[0]['lr'], .001)
                if arm == 'mw_rank':
                    self.assertEqual(chemical.rank_delta, 0)
                    self.assertEqual(chemical.teacher_temperature, 0)
                    self.assertEqual(chemical.teacher_normalization, 'none')
                    self.assertFalse(chemical.clip_targets)
                    paired = torch.tensor([1., 1., 0.], requires_grad=True)
                    value = chemical._per_sequence_rank_loss(torch.tensor([-.2, .1, 1.5]), paired)
                    self.assertAlmostEqual(float(value), 0.)
                    self.assertTrue(torch.isfinite(torch.autograd.grad(value, paired)[0]).all())

    def test_dispatch_includes_registered_mw_arm_and_each_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'plan.json').write_text(json.dumps({
                'pretrain_epochs': 10, 'finetune_epochs': 5, 'arms': ['mw_rank']}))
            run = root/'runs/pretrain'
            run.mkdir(parents=True)
            (run/'epoch-010.pt').touch()
            mw = root/'runs/mw_rank'
            mw.mkdir()
            for epoch in range(1, 6):
                (mw/f'epoch-{epoch:03d}.pt').touch()
            with patch('scripts.efold_hf.workflow.submit_once') as submit:
                dispatch_completed(root, 'pretrain')
                submit.assert_any_call(root, 'start-mw_rank', 'work', 'mw_rank')
                dispatch_completed(root, 'mw_rank')
                for epoch in range(1, 6):
                    submit.assert_any_call(root, f'eval-mw_rank-{epoch}', 'evaluate', 'mw_rank', epoch)

    def test_transition_uses_configured_pretrain_epoch(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'plan.json').write_text(json.dumps({'pretrain_epochs': 10, 'finetune_epochs': 5}))
            run = root/'runs/pretrain'
            run.mkdir(parents=True)
            (run/'epoch-100.pt').touch()
            with patch('scripts.efold_hf.workflow.submit_once') as submit:
                dispatch_completed(root, 'pretrain')
                submit.assert_not_called()
                (run/'epoch-010.pt').touch()
                dispatch_completed(root, 'pretrain')
                self.assertEqual(submit.call_count, 5)
                submit.assert_any_call(root, 'eval-pretrain-10', 'evaluate', 'pretrain', 10)
                for arm in ('rank', 'nll', 'shape_fy', 'assisted'):
                    submit.assert_any_call(root, 'start-'+arm, 'work', arm)

    def test_worker_resumes_mid_epoch_and_stops_at_ten(self):
        from pytorch_optimizer import AdaBelief
        plan = {'pretrain_epochs': 10, 'finetune_epochs': 5, 'worker_hours': 20}
        seed_all(1234)
        model = torch.nn.Linear(1, 1)
        optimizer = AdaBelief(model.parameters(), lr=.001)
        for _ in range(5):
            backward_step(model(torch.ones(1)).square().mean(), model, optimizer)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'preflight.json').write_text('{"success":true}')
            save_checkpoint(root/'runs/pretrain/latest.pt', model, optimizer, None,
                            {'epoch': 3, 'cursor': 1, 'updates': 5, 'loss_sum': 1.})
            resumed_model = torch.nn.Linear(1, 1)
            resumed_optimizer = AdaBelief(resumed_model.parameters(), lr=.001)
            def loss(*_):
                return resumed_model(torch.ones(1)).square().mean()
            with patch('scripts.efold_hf.workflow.verify', return_value=plan), \
                 patch('scripts.efold_hf.workflow.build', return_value=(resumed_model, resumed_optimizer, None, None, None, 'cpu')), \
                 patch('scripts.efold_hf.workflow.read_jsonl', return_value=[0, 1]), \
                 patch('scripts.efold_hf.workflow.sequence_loss', side_effect=loss), \
                 patch('scripts.efold_hf.workflow.dispatch_completed'):
                train_worker(root, 'pretrain', -1)
            final = torch.load(root/'runs/pretrain/epoch-010.pt', weights_only=False)
            self.assertEqual(final['state']['epoch'], 10)
            self.assertEqual(final['state']['updates'], 20)
            self.assertEqual(json.loads((root/'runs/pretrain/complete.json').read_text())['epochs'], 10)
            self.assertFalse((root/'runs/pretrain/epoch-100.pt').exists())

    def test_full_chemical_coverage_and_twenty_percent_replay(self):
        slots = schedule(100, 19, 2)
        self.assertEqual(len(slots), 125)
        self.assertEqual(sorted(i for kind, i in slots if kind == 'primary'), list(range(100)))
        self.assertEqual(sum(kind == 'replay' for kind, _ in slots), 25)
        self.assertEqual(slots, schedule(100, 19, 2))
        self.assertNotEqual(slots, schedule(100, 19, 3))
        self.assertEqual(slots[39:], schedule(100, 19, 2)[39:])
        self.assertEqual(len(schedule(100, 19, 2, replay=False)), 100)

    def test_observation_masks_preserve_negative_signal(self):
        row = {'sequence': 'AACGUA', 'dms': [-1000., -.2, .5, .6, .7, float('nan')],
               'shape': [-1000., -.2, .5, .6, .7, None]}
        obs = clean_observations(row)
        self.assertEqual(obs[0]['values'], [-999., -.2, .5, -999., -999., -999.])
        self.assertEqual(obs[1]['values'], [-999., -.2, .5, .6, .7, -999.])

    def test_noncrossing_projection_matches_exhaustive_optimum(self):
        sequence = 'GGGGAAAACCCC'
        candidates = [(0, 8), (1, 11), (2, 10), (3, 9)]
        result = project_pairs(sequence, candidates)
        best = 0
        for k in range(len(candidates)+1):
            for subset in itertools.combinations(candidates, k):
                if all(not (i < a < j < b or a < i < b < j)
                       for (i, j), (a, b) in itertools.combinations(subset, 2)):
                    best = max(best, k)
        self.assertEqual(len(result), best)
        self.assertTrue(set(result) <= set(candidates))

    def test_sparse_probabilities_charge_both_bases(self):
        native = [[], [(4, .8), (3, .1)], [(3, .2)], [], []]
        np.testing.assert_allclose(unpaired_probabilities(native, 4), [.1, .8, .7, .2])

    def test_pair_metrics_keep_crossing_reference_pairs(self):
        metrics = pair_metrics([(0, 7), (3, 9)], [(0, 7)])
        self.assertEqual(metrics['tp'], 1)
        self.assertEqual(metrics['fn'], 1)
        self.assertAlmostEqual(metrics['f1'], 2/3)

    def test_resume_matches_uninterrupted_optimizer_and_dropout(self):
        from pytorch_optimizer import AdaBelief
        seed_all(1234)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(.5))
        optimizer = AdaBelief(model.parameters(), lr=.001)
        x = torch.ones(4)
        backward_step(model(x).square().mean(), model, optimizer)
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp) / 'state.pt'
            save_checkpoint(checkpoint, model, optimizer, None, {'epoch': 2, 'cursor': 7})
            backward_step(model(x).square().mean(), model, optimizer)
            expected = {k: v.clone() for k, v in model.state_dict().items()}
            state = restore(checkpoint, model, optimizer)
            self.assertEqual(state['cursor'], 7)
            backward_step(model(x).square().mean(), model, optimizer)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), key)


if __name__ == '__main__':
    unittest.main()
