from __future__ import annotations

import logging
import os
import random
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Optional, cast

#import numpy as np
import torch
import torch.backends.cudnn
#import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from dataset import FastaDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import BPseqDataset
from .fold.fold import AbstractFold
from .fold.linearfold import LinearFold
from .fold.linearfold2d import LinearFold2D
from .fold.linearfoldv import LinearFoldV
from .fold.mix import MixedFold
from .fold.mix1d import MixedFold1D
from .fold.mix_bl import MixedFoldBL
from .fold.mix_linearfold import MixedLinearFold
from .fold.mix_linearfold1d import MixedLinearFold1D
from .fold.mix_linearfold2d import MixedLinearFold2D
from .fold.rnafold import RNAFold
from .fold.zuker import ZukerFold
from .fold.zuker_bl import ZukerFoldBL
from .loss import StructuredLoss, StructuredLossWithTurner

try:
    from torch.utils.tensorboard.writer import SummaryWriter
except ImportError:
    pass


class Train:
    step: int = 0
    train_loader: Optional[DataLoader[tuple[str, str, torch.Tensor]]]
    test_loader: Optional[DataLoader[tuple[str, str, torch.Tensor]]]
    optimizer: optim.Optimizer
    model: AbstractFold
    loss_fn: StructuredLoss | StructuredLossWithTurner
    disable_progress_bar: bool
    writer: Optional[SummaryWriter]

    def __init__(self):
        self.train_loader = None
        self.test_loader = None


    def train(self, epoch: int) -> None:
        if self.train_loader is None:
            raise RuntimeError
        self.model.train()
        n_dataset = len(cast(FastaDataset, self.train_loader.dataset))
        loss_total, num = 0., 0
        running_loss, n_running_loss = 0, 0
        start = time.time()
        with tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, pairs in self.train_loader:
                logging.info(f"Step: {self.step}, {fnames}")
                self.step += 1
                n_batch = len(seqs)
                self.optimizer.zero_grad()
                loss = torch.sum(self.loss_fn(seqs, pairs, fname=fnames))
                # if float(loss.item()) < 0.:
                #     next
                # FIXME: loss must be positive, but this is not guaranteed for LinearFold model. Any idea?
                loss_total += loss.item()
                num += n_batch
                loss.backward()
                # if self.verbose:
                #     for n, p in self.model.named_parameters():
                #         print(n, torch.min(p).item(), torch.max(p).item(), 
                #             torch.min(cast(torch.Tensor, p.grad)).item(), 
                #             torch.max(cast(torch.Tensor, p.grad)).item())
                self.optimizer.step()

                pbar.set_postfix(train_loss='{:.3e}'.format(loss_total / num))
                pbar.update(n_batch)

                running_loss += loss.item()
                n_running_loss += n_batch
                if n_running_loss >= 100 or num >= n_dataset:
                    running_loss /= n_running_loss
                    if self.writer is not None:
                        self.writer.add_scalar("train/loss", running_loss, (epoch-1) * n_dataset + num)
                    running_loss, n_running_loss = 0, 0
        elapsed_time = time.time() - start
        print('Train Epoch: {}\tLoss: {:.6f}\tTime: {:.3f}s'.format(epoch, loss_total / num, elapsed_time))


    def test(self, epoch: int) -> None:
        if self.test_loader is None:
            raise RuntimeError
        self.model.eval()
        n_dataset = len(cast(FastaDataset, self.test_loader.dataset))
        loss_total, num = 0, 0
        start = time.time()
        with torch.no_grad(), tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, pairs in self.test_loader:
                n_batch = len(seqs)
                loss = self.loss_fn(seqs, pairs, fname=fnames)
                loss_total += loss.item()
                num += n_batch
                pbar.set_postfix(test_loss='{:.3e}'.format(loss_total / num))
                pbar.update(n_batch)

        elapsed_time = time.time() - start
        if self.writer is not None:
            self.writer.add_scalar("test/loss", loss_total / num, epoch * n_dataset)
            self.writer.add_scalar("test/loss", loss_total / num, epoch * n_dataset)
        print('Test Epoch: {}\tLoss: {:.6f}\tTime: {:.3f}s'.format(epoch, loss_total / num, elapsed_time))


    def save_checkpoint(self, outdir, epoch):
        filename = os.path.join(outdir, 'epoch-{}'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)


    def resume_checkpoint(self, filename: str) -> int:
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return epoch


    def build_model(self, args: Namespace) -> tuple[AbstractFold, dict[str, Any]]:
        if args.model == 'Turner':
            return RNAFold(), {}

        if args.model == 'LinearFoldV':
            return LinearFoldV(), {}

        config = {
            'max_helix_length': args.max_helix_length,
            'embed_size' : args.embed_size,
            'num_filters': args.num_filters if args.num_filters is not None else (96,),
            'filter_size': args.filter_size if args.filter_size is not None else (5,),
            'pool_size': args.pool_size if args.pool_size is not None else (1,),
            'dilation': args.dilation, 
            'num_lstm_layers': args.num_lstm_layers, 
            'num_lstm_units': args.num_lstm_units,
            'num_transformer_layers': args.num_transformer_layers,
            'num_transformer_hidden_units': args.num_transformer_hidden_units,
            'num_transformer_att': args.num_transformer_att,
            'num_hidden_units': args.num_hidden_units if args.num_hidden_units is not None else (32,),
            'num_paired_filters': args.num_paired_filters,
            'paired_filter_size': args.paired_filter_size,
            'dropout_rate': args.dropout_rate,
            'fc_dropout_rate': args.fc_dropout_rate,
            'num_att': args.num_att,
            'pair_join': args.pair_join,
            'no_split_lr': args.no_split_lr,
            'bl_size': args.bl_size,
            'sym_opts': args.paired_opt,
        }

        if args.model == 'Zuker':
            model = ZukerFold(model_type='M', **config)

        elif args.model == 'ZukerC':
            model = ZukerFold(model_type='C', **config)

        elif args.model == 'ZukerL':
            model = ZukerFold(model_type="L", **config)

        elif args.model == 'ZukerS':
            model = ZukerFold(model_type="S", **config)

        elif args.model == 'Mix':
            from . import param_turner2004
            model = MixedFold(init_param=param_turner2004, **config)

        elif args.model == 'MixC':
            from . import param_turner2004
            model = MixedFold(init_param=param_turner2004, model_type='C', **config)

        elif args.model == 'Mix1D':
            from . import param_turner2004
            model = MixedFold1D(init_param=param_turner2004, **config)

        elif args.model == 'ZukerBL':
            model = ZukerFoldBL(**config)

        elif args.model == 'MixedBL':
            from . import param_turner2004
            model = MixedFoldBL(init_param=param_turner2004, **config)

        elif args.model == 'LinearFold':
            model = LinearFold(**config)

        elif args.model == 'MixedLinearFold':
            from . import param_turner2004
            model = MixedLinearFold(init_param=param_turner2004, **config)

        elif args.model == 'LinearFold2D':
            model = LinearFold2D(**config)

        elif args.model == 'MixedLinearFold2D':
            from . import param_turner2004
            model = MixedLinearFold2D(init_param=param_turner2004, **config)

        elif args.model == 'MixedLinearFold1D':
            from . import param_turner2004
            model = MixedLinearFold1D(init_param=param_turner2004, **config)

        else:
            raise(RuntimeError('not implemented'))

        return model, config


    def build_optimizer(self, optimizer: str, model: AbstractFold, lr: float, l2_weight: float) -> optim.Optimizer:
        if optimizer == 'Adam':
            return optim.Adam(model.parameters(), lr=lr, amsgrad=False, weight_decay=l2_weight)
        elif optimizer =='AdamW':
            return optim.AdamW(model.parameters(), lr=lr, amsgrad=False, weight_decay=l2_weight)
        elif optimizer == 'RMSprop':
            return optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer == 'SGD':
            return optim.SGD(model.parameters(), nesterov=True, lr=lr, momentum=0.9, weight_decay=l2_weight)
            #return optim.SGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer == 'ASGD':
            return optim.ASGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        else:
            raise(RuntimeError('not implemented'))


    def build_loss_function(self, loss_func: str, model: AbstractFold, args: Namespace) -> StructuredLoss | StructuredLossWithTurner:
        if loss_func == 'hinge':
            return StructuredLoss(model, 
                            loss_pos_paired=args.loss_pos_paired, loss_neg_paired=args.loss_neg_paired, 
                            loss_pos_unpaired=args.loss_pos_unpaired, loss_neg_unpaired=args.loss_neg_unpaired, 
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight)
        if loss_func == 'hinge_mix':
            return StructuredLossWithTurner(model,
                            loss_pos_paired=args.loss_pos_paired, loss_neg_paired=args.loss_neg_paired, 
                            loss_pos_unpaired=args.loss_pos_unpaired, loss_neg_unpaired=args.loss_neg_unpaired, 
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight, sl_weight=args.score_loss_weight)
        else:
            raise(RuntimeError('not implemented'))


    def save_config(self, file: str, config: dict[str, Any]) -> None:
        with open(file, 'w') as f:
            for k, v in config.items():
                k = '--' + k.replace('_', '-')
                if type(v) is bool: # pylint: disable=unidiomatic-typecheck
                    if v:
                        f.write('{}\n'.format(k))
                elif isinstance(v, list) or isinstance(v, tuple):
                    for vv in v:
                        f.write('{}\n{}\n'.format(k, vv))
                else:
                    f.write('{}\n{}\n'.format(k, v))


    def run(self, args: Namespace, conf: Optional[str] = None) -> None:
        self.disable_progress_bar = args.disable_progress_bar
        loglevel = 'INFO' if args.verbose else args.loglevel
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=getattr(logging, loglevel, None))
        self.writer = None
        if args.log_dir is not None and 'SummaryWriter' in globals():
            self.writer = SummaryWriter(log_dir=args.log_dir)

        train_dataset = BPseqDataset(args.input)
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True) # works well only for batch_size=1!!
        if args.test_input is not None:
            test_dataset = BPseqDataset(args.test_input)
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # works well only for batch_size=1!!

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.model, config = self.build_model(args)
        config.update({ 'model': args.model, 'param': args.param })
        
        if args.init_param != '':
            init_param = Path(args.init_param)
            if not init_param.exists() and conf is not None:
                init_param = Path(conf) / init_param
            p = torch.load(init_param)
            if isinstance(p, dict) and 'model_state_dict' in p:
                p = p['model_state_dict']
            self.model.load_state_dict(p)

        if args.gpu >= 0:
            self.model.to(torch.device("cuda", args.gpu))
        self.optimizer = self.build_optimizer(args.optimizer, self.model, args.lr, args.l2_weight)
        self.loss_fn = self.build_loss_function(args.loss_func, self.model, args)

        checkpoint_epoch = 0
        if args.resume is not None:
            checkpoint_epoch = self.resume_checkpoint(args.resume)

        for epoch in range(checkpoint_epoch+1, args.epochs+1):
            self.train(epoch)
            if self.test_loader is not None:
                self.test(epoch)
            if args.log_dir is not None:
                self.save_checkpoint(args.log_dir, epoch)

        if args.param is not None:
            torch.save(self.model.state_dict(), args.param)
        if args.save_config is not None:
            self.save_config(args.save_config, config)

        #return self.model


    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('train', help='training')
        # input
        subparser.add_argument('input', type=str,
                            help='Training data of the list of BPSEQ-formatted files')
        subparser.add_argument('--test-input', type=str,
                            help='Test data of the list of BPSEQ-formatted files')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--param', type=str, default='param.pth',
                            help='output file name of trained parameters')
        subparser.add_argument('--init-param', type=str, default='',
                            help='the file name of the initial parameters')

        gparser = subparser.add_argument_group("Training environment")
        subparser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        subparser.add_argument('--log-dir', type=str, default=None,
                            help='Directory for storing logs')
        subparser.add_argument('--resume', type=str, default=None,
                            help='Checkpoint file for resume')
        subparser.add_argument('--save-config', type=str, default=None,
                            help='save model configurations')
        subparser.add_argument('--disable-progress-bar', action='store_true',
                            help='disable the progress bar in training')
        subparser.add_argument('--verbose', action='store_true',
                            help='enable verbose outputs for debugging')
        subparser.add_argument('--loglevel', choices=('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'),
                            default='WARNING', help="set the log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')")

        gparser = subparser.add_argument_group("Optimizer setting")
        gparser.add_argument('--optimizer', choices=('Adam', 'AdamW', 'RMSprop', 'SGD', 'ASGD'), default='AdamW')
        gparser.add_argument('--l1-weight', type=float, default=0.,
                            help='the weight for L1 regularization (default: 0)')
        gparser.add_argument('--l2-weight', type=float, default=0.,
                            help='the weight for L2 regularization (default: 0)')
        gparser.add_argument('--score-loss-weight', type=float, default=1.,
                            help='the weight for score loss for hinge_mix loss (default: 1)')
        gparser.add_argument('--lr', type=float, default=0.001,
                            help='the learning rate for optimizer (default: 0.001)')
        gparser.add_argument('--loss-func', choices=('hinge', 'hinge_mix'), default='hinge',
                            help="loss fuction ('hinge', 'hinge_mix') ")
        gparser.add_argument('--loss-pos-paired', type=float, default=0.5,
                            help='the penalty for positive base-pairs for loss augmentation (default: 0.5)')
        gparser.add_argument('--loss-neg-paired', type=float, default=0.005,
                            help='the penalty for negative base-pairs for loss augmentation (default: 0.005)')
        gparser.add_argument('--loss-pos-unpaired', type=float, default=0,
                            help='the penalty for positive unpaired bases for loss augmentation (default: 0)')
        gparser.add_argument('--loss-neg-unpaired', type=float, default=0,
                            help='the penalty for negative unpaired bases for loss augmentation (default: 0)')

        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--model', choices=('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'ZukerC', 'Mix', 'MixC', 'Mix1D', 'LinearFold', 'LinearFoldV', 'MixedLinearFold', 'ZukerBL', 'MixedBL', 'LinearFold2D', 'MixedLinearFold2D', 'MixedLinearFold1D'), default='Turner', 
                        help="Folding model ('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'ZukerC', 'Mix', 'MixC', 'Mix1D', 'LinearFold', 'LinearFoldV', 'MixedLinearFold', 'ZukerBL', 'MixedBL', 'LinearFold2D', 'MixedLinearFold2D', 'MixedLinearFold1D')")
        gparser.add_argument('--max-helix-length', type=int, default=30, 
                        help='the maximum length of helices (default: 30)')
        gparser.add_argument('--embed-size', type=int, default=0,
                        help='the dimention of embedding (default: 0 == onehot)')
        gparser.add_argument('--num-filters', type=int, action='append',
                        help='the number of CNN filters (default: 96)')
        gparser.add_argument('--filter-size', type=int, action='append',
                        help='the length of each filter of CNN (default: 5)')
        gparser.add_argument('--pool-size', type=int, action='append',
                        help='the width of the max-pooling layer of CNN (default: 1)')
        gparser.add_argument('--dilation', type=int, default=0, 
                        help='Use the dilated convolution (default: 0)')
        gparser.add_argument('--num-lstm-layers', type=int, default=0,
                        help='the number of the LSTM hidden layers (default: 0)')
        gparser.add_argument('--num-lstm-units', type=int, default=0,
                        help='the number of the LSTM hidden units (default: 0)')
        gparser.add_argument('--num-transformer-layers', type=int, default=0,
                        help='the number of the transformer layers (default: 0)')
        gparser.add_argument('--num-transformer-hidden-units', type=int, default=2048,
                        help='the number of the hidden units of each transformer layer (default: 2048)')
        gparser.add_argument('--num-transformer-att', type=int, default=8,
                        help='the number of the attention heads of each transformer layer (default: 8)')
        gparser.add_argument('--num-paired-filters', type=int, action='append', default=[],
                        help='the number of CNN filters (default: 96)')
        gparser.add_argument('--paired-filter-size', type=int, action='append', default=[],
                        help='the length of each filter of CNN (default: 5)')
        gparser.add_argument('--num-hidden-units', type=int, action='append',
                        help='the number of the hidden units of full connected layers (default: 32)')
        gparser.add_argument('--dropout-rate', type=float, default=0.0,
                        help='dropout rate of the CNN and LSTM units (default: 0.0)')
        gparser.add_argument('--fc-dropout-rate', type=float, default=0.0,
                        help='dropout rate of the hidden units (default: 0.0)')
        gparser.add_argument('--num-att', type=int, default=0,
                        help='the number of the heads of attention (default: 0)')
        gparser.add_argument('--pair-join', choices=('cat', 'add', 'mul', 'bilinear'), default='cat', 
                            help="how pairs of vectors are joined ('cat', 'add', 'mul', 'bilinear') (default: 'cat')")
        gparser.add_argument('--no-split-lr', default=False, action='store_true')
        gparser.add_argument('--bl-size', type=int, default=4,
                        help='the input dimension of the bilinear layer of LinearFold model (default: 4)')
        gparser.add_argument('--paired-opt', choices=('0_1_1', 'fixed', 'symmetric'), default='0_1_1')

        subparser.set_defaults(func = lambda args, conf: Train().run(args, conf))
