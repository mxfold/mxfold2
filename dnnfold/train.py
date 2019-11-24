import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .dataset import BPseqDataset
from .fold.nussinov import NussinovFold
from .fold.rnafold import RNAFold
from .fold.zuker import ZukerFold


class StructuredLoss(nn.Module):
    def __init__(self, model, loss_pos_paired=0, loss_neg_paired=0, loss_pos_unpaired=0, loss_neg_unpaired=0, 
                l1_weight=0., l2_weight=0., verbose=False):
        super(StructuredLoss, self).__init__()
        self.model = model
        self.loss_pos_paired = loss_pos_paired
        self.loss_neg_paired = loss_neg_paired
        self.loss_pos_unpaired = loss_pos_unpaired
        self.loss_neg_unpaired = loss_neg_unpaired
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.verbose = verbose


    def forward(self, seq, pair, fname=None):
        pred, pred_s, _ = self.model(seq, reference=pair,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = self.model(seq, constraint=pair, max_internal_length=None)
        loss = pred - ref
        if self.verbose:
            print("Loss = {} = ({} - {})".format(loss.item(), pred.item(), ref.item()))
            print(seq)
            print(pred_s)
            print(ref_s)
        if loss.item()> 1e10 or torch.isnan(loss):
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)
            print(pair)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))

        # if self.l2_weight > 0.0:
        #     l2_reg = 0.0
        #     for p in self.model.parameters():
        #         l2_reg += torch.sum((self.l2_weight * p) ** 2)
        #     loss += torch.sqrt(l2_reg)

        return loss


class Train:
    step = 0

    def __init__(self):
        self.train_loader = None
        self.test_loader = None


    def train(self, epoch):
        self.model.train()
        n_dataset = len(self.train_loader.dataset)
        loss_total, num = 0, 0
        running_loss, n_running_loss = 0, 0
        with tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, pairs, _ in self.train_loader:
                if self.verbose:
                    print()
                    print("Step: {}, {}".format(self.step, fnames))
                    self.step += 1
                n_batch = len(seqs)
                self.optimizer.zero_grad()
                loss = torch.sum(self.loss_fn(seqs, pairs, fname=fnames))
                loss_total += loss.item()
                num += n_batch
                loss.backward()
                if self.verbose:
                    for n, p in self.model.named_parameters():
                        print(n, torch.min(p).item(), torch.max(p).item(), torch.min(p.grad).item(), torch.max(p.grad).item())
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
        if self.verbose:
            print()
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss_total / num))


    def test(self, epoch):
        self.model.eval()
        n_dataset = len(self.test_loader.dataset)
        loss_total, num = 0, 0
        with torch.no_grad(), tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, pairs, _ in self.test_loader:
                n_batch = len(seqs)
                loss = self.loss_fn(seqs, pairs, fname=fnames)
                loss_total += loss.item()
                num += n_batch
                pbar.set_postfix(test_loss='{:.3e}'.format(loss_total / num))
                pbar.update(n_batch)

        if self.writer is not None:
            self.writer.add_scalar("test/loss", epoch * n_dataset, loss_total / num)
        print('Test Epoch: {}\tLoss: {:.6f}'.format(epoch, loss_total / num))


    def save_checkpoint(self, outdir, epoch):
        filename = os.path.join(outdir, 'epoch-{}'.format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filename)


    def resume_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return epoch


    def build_model(self, args):
        if args.model == 'Turner':
            return RNAFold(), {}

        config = {
            'embed_size' : args.embed_size,
            'num_filters': args.num_filters if args.num_filters is not None else (96,),
            'filter_size': args.filter_size if args.filter_size is not None else (5,),
            'pool_size': args.pool_size if args.pool_size is not None else (1,),
            'dilation': args.dilation, 
            'num_lstm_layers': args.num_lstm_layers, 
            'num_lstm_units': args.num_lstm_units,
            'num_hidden_units': args.num_hidden_units if args.num_hidden_units is not None else (32,),
            'dropout_rate': args.dropout_rate,
            'fc_dropout_rate': args.fc_dropout_rate,
            'lstm_cnn': args.lstm_cnn,
            'num_att': args.num_att,
            'context_length': args.context_length,
            'mix_base': args.mix_base,
            'pair_join': args.pair_join,
            'fc': args.fc,
            'no_split_lr': args.no_split_lr
        }

        if args.model == 'Zuker':
            model = ZukerFold(model_type='M', **config)

        elif args.model == 'ZukerL':
            model = ZukerFold(model_type="L", **config)

        elif args.model == 'ZukerS':
            model = ZukerFold(model_type="S", **config)

        elif args.model == 'ZukerP':
            model = ZukerFold(model_type="P", **config)

        elif args.model == 'Nussinov':
            model = NussinovFold(**config)

        elif args.model == 'NussinovP':
            model = NussinovFold(model_type='P', **config)

        else:
            raise('not implemented')

        return model, config


    def build_optimizer(self, optimizer, model, lr, l2_weight):
        if optimizer == 'Adam':
            return optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=l2_weight)
        elif optimizer =='AdamW':
            return optim.AdamW(model.parameters(), lr=lr, amsgrad=True, weight_decay=l2_weight)
        elif optimizer == 'RMSprop':
            return optim.RMSprop(model.parameters(), lr=lr, weight_decay=l2_weight)
        elif optimizer == 'SGD':
            return optim.SGD(model.parameters(), nesterov=True, lr=lr, momentum=0.9, weight_decay=l2_weight)
        elif optimizer == 'ASGD':
            return optim.ASGD(model.parameters(), lr=lr, weight_decay=l2_weight)
        else:
            raise('not implemented')

    
    def save_config(self, file, config):
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


    def run(self, args):
        self.disable_progress_bar = args.disable_progress_bar
        self.verbose = args.verbose
        self.writer = None
        if args.log_dir is not None:
            self.writer = SummaryWriter(log_dir=args.log_dir)

        train_dataset = BPseqDataset(args.input, unpaired='x')
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        if args.test_input is not None:
            test_dataset = BPseqDataset(args.test_input, unpaired='x')
            self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)

        self.model, config = self.build_model(args)
        config.update({ 'model': args.model, 'param': args.param })
        
        if args.gpu >= 0:
            self.model.to(torch.device("cuda", args.gpu))
        self.optimizer = self.build_optimizer(args.optimizer, self.model, args.lr, args.l2_weight)
        self.loss_fn = StructuredLoss(self.model, verbose=self.verbose,
                            loss_pos_paired=args.loss_pos_paired, loss_neg_paired=args.loss_neg_paired, 
                            loss_pos_unpaired=args.loss_pos_unpaired, loss_neg_unpaired=args.loss_neg_unpaired, 
                            l1_weight=args.l1_weight, l2_weight=args.l2_weight)

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

        return self.model


    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('train', help='training')
        # input
        subparser.add_argument('input', type=str,
                            help='Training data of BPSEQ-formatted file')
        subparser.add_argument('--test-input', type=str,
                            help='Test data of BPSEQ-formatted file')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--param', type=str, default='param.pth',
                            help='output file name of trained parameters')

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

        gparser = subparser.add_argument_group("Optimizer setting")
        gparser.add_argument('--optimizer', choices=('Adam', 'AdamW', 'RMSprop', 'SGD', 'ASGD'), default='AdamW')
        gparser.add_argument('--l1-weight', type=float, default=0.,
                            help='the weight for L1 regularization (default: 0)')
        gparser.add_argument('--l2-weight', type=float, default=0.,
                            help='the weight for L2 regularization (default: 0)')
        gparser.add_argument('--lr', type=float, default=0.01,
                            help='the learning rate for optimizer (default: 0.01)')
        gparser.add_argument('--loss-pos-paired', type=float, default=1,
                            help='the penalty for positive base-pairs for loss augmentation (default: 1)')
        gparser.add_argument('--loss-neg-paired', type=float, default=1,
                            help='the penalty for negative base-pairs for loss augmentation (default: 1)')
        gparser.add_argument('--loss-pos-unpaired', type=float, default=1,
                            help='the penalty for positive unpaired bases for loss augmentation (default: 1)')
        gparser.add_argument('--loss-neg-unpaired', type=float, default=1,
                            help='the penalty for negative unpaired bases for loss augmentation (default: 1)')

        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--model', choices=('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'ZukerP', 'Nussinov', 'NussinovP'), default='Turner', 
                            help="Folding model ('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'ZukerP', 'Nussinov', 'NussinovP')")
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
        gparser.add_argument('--num-hidden-units', type=int, action='append',
                        help='the number of the hidden units of full connected layers (default: 32)')
        gparser.add_argument('--dropout-rate', type=float, default=0.0,
                        help='dropout rate of the CNN and LSTM units (default: 0.0)')
        gparser.add_argument('--fc-dropout-rate', type=float, default=0.0,
                        help='dropout rate of the hidden units (default: 0.0)')
        gparser.add_argument('--lstm-cnn', default=False, action='store_true',
                        help='use LSTM layer before CNN (default: False)')
        gparser.add_argument('--num-att', type=int, default=0,
                        help='the number of the heads of attention (default: 0)')
        gparser.add_argument('--context-length', type=int, default=1,
                        help='the length of context for FC layers (default: 1)')
        gparser.add_argument('--mix-base', default=0, type=int, 
                        help='the length of context for mixing the base features to the input of the folding layer (default: 0)')
        gparser.add_argument('--pair-join', choices=('cat', 'add', 'mul', 'bilinear'), default='cat', 
                            help="how pairs of vectors are joined ('cat', 'add', 'mul', 'bilinear') (default: 'cat')")
        gparser.add_argument('--fc', choices=('linear', 'conv'), default='linear', 
                            help="type of final layers ('linear', 'conv') (default: 'linear')")
        gparser.add_argument('--no-split-lr', default=False, action='store_true')

        subparser.set_defaults(func = lambda args: Train().run(args))
