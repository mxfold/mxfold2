"""Regression tests for fixed Turner scores with real mixed-model MAP folds."""
import argparse
import unittest

import torch

from mxfold2 import param_turner2004
from mxfold2.fold.mix import MixedFold
from mxfold2.fold.shape_layers import Wu
from mxfold2.loss.shape_nll_loss import ShapeNLLLoss
from mxfold2.loss.shape_rank_loss import ShapeRankLoss
from mxfold2.train import Train


def small_model():
    return MixedFold(
        init_param=param_turner2004, embed_size=4,
        num_filters=(4,), filter_size=(3,), pool_size=(1,),
        num_hidden_units=(4,), num_paired_filters=(), paired_filter_size=(),
        dropout_rate=0., fc_dropout_rate=0., additional_params=True,
        paired_opt='symmetric',
    )


class IMLETurnerDefaultTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.original_threads)

    def check_update(self, kind, *, train_turner=False, **options):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = small_model()
            shape_model = Wu()
            kwargs = dict(perturb=.1, nu=10., normalization='rms',
                          difference_mode='central')
            kwargs.update(options)
            # Omit the keyword in default cases to test the public defaults.
            if train_turner:
                kwargs['train_turner'] = True
            if kind == 'nll':
                loss_fn = ShapeNLLLoss(model, [shape_model], **kwargs)
            else:
                loss_fn = ShapeRankLoss(model, objective=kind, **kwargs)
            before_turner = {k: v.detach().clone() for k, v in model.turner.state_dict().items()}
            before_neural = {k: v.detach().clone() for k, v in model.zuker.named_parameters()}
            # Include Turner in the optimizer deliberately: a zero gradient
            # would still permit AdamW weight decay, whereas grad=None does not.
            optimizer = torch.optim.AdamW(
                list(model.parameters()) + list(shape_model.parameters()),
                lr=.001, weight_decay=.01,
            )
            sequence = 'GCGCGCGCGCGCGCGCGCGC'
            targets = torch.tensor([-999.] + [.8] * 8 + [.05] * 4 + [.8] * 8)
            loss = loss_fn([sequence], [targets]).sum()
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            if train_turner:
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in model.turner.parameters()))
            else:
                self.assertTrue(all(p.requires_grad and p.grad is None
                                    for p in model.turner.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.zuker.parameters()))
            if kind == 'nll':
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                    for p in shape_model.parameters()))
            optimizer.step()
            changed = any(not torch.equal(v, before_turner[k])
                          for k, v in model.turner.state_dict().items())
            self.assertEqual(changed, train_turner)
            self.assertTrue(any(not torch.equal(v, before_neural[k])
                                for k, v in model.zuker.named_parameters()))

    def test_default_keeps_turner_fixed_with_regularization_and_pseudo_fy(self):
        for kind in ('nll', 'pairwise_logistic', 'mann_whitney'):
            for regularization in (0., .125):
                with self.subTest(kind=kind, regularization=regularization):
                    self.check_update(kind, sl_weight=regularization,
                                      pseudo_fy_weight=.2 if regularization else 0.)

    def test_forward_and_multiple_independent_samples_keep_turner_fixed(self):
        for kind in ('nll', 'pairwise_logistic', 'mann_whitney'):
            with self.subTest(kind=kind):
                self.check_update(kind, difference_mode='forward', num_samples=2)

    def test_mean_paired_sampling_keeps_turner_fixed(self):
        for kind in ('pairwise_logistic', 'mann_whitney'):
            with self.subTest(kind=kind):
                self.check_update(kind, num_samples=2, sample_aggregation='mean_paired')

    def test_legacy_turner_updates_require_explicit_opt_in(self):
        for kind in ('nll', 'pairwise_logistic', 'mann_whitney'):
            with self.subTest(kind=kind):
                self.check_update(kind, train_turner=True)

    def test_turner_energies_still_contribute_to_structure_scores(self):
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(0)
            model = small_model().eval()
            sequence = 'GGGAAACCC'
            pairs = [[0, 9, 8, 7, 0, 0, 0, 3, 2, 1]]
            initial, _, _ = model([sequence], constraint=pairs)
            model.turner.score_stack.add_(1.)
            changed, _, _ = model([sequence], constraint=pairs)
            self.assertGreater(float(changed - initial), 0.)

    def test_explicitly_frozen_run_matches_legacy_neural_update(self):
        # The already-running experiment explicitly freezes Turner. Removing
        # its unused gradient inputs must preserve the neural update exactly.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = small_model()
            model.turner.requires_grad_(False)
            initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
            states, values = [], []
            for legacy in (False, True):
                model.load_state_dict(initial)
                model.zero_grad(set_to_none=True)
                loss_fn = ShapeRankLoss(
                    model, objective='mann_whitney', train_turner=legacy,
                    perturb=.1, nu=.1, normalization='rms',
                    difference_mode='central', sl_weight=.125,
                )
                optimizer = torch.optim.AdamW(
                    (p for p in model.parameters() if p.requires_grad),
                    lr=1e-4, weight_decay=1e-4,
                )
                torch.manual_seed(123)
                loss = loss_fn(['GCGCGCGCGCGCGCGCGCGC'],
                               [torch.tensor([-999.] + [.05, .8] * 10)])
                values.append(loss.detach().clone())
                loss.backward()
                optimizer.step()
                states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
            for key in states[0]:
                self.assertTrue(torch.equal(states[0][key], states[1][key]), key)

    def test_cli_and_loss_factory_default_and_opt_in(self):
        parser = argparse.ArgumentParser()
        Train.add_args(parser.add_subparsers())
        for enabled in (False, True):
            command = ['train', 'unused.lst']
            if enabled:
                command.append('--shape-imle-train-turner')
            args = parser.parse_args(command)
            self.assertEqual(args.shape_imle_train_turner, enabled)
            for kind in ('shape_nll', 'shape_rank'):
                with self.subTest(enabled=enabled, kind=kind):
                    loss_fn = Train().build_shape_loss_function(kind, small_model(), args, [Wu()])
                    self.assertEqual(loss_fn.train_turner, enabled)


if __name__ == '__main__':
    unittest.main()
