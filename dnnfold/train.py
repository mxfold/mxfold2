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
from .fold.rnafold import RNAFold
from .fold.positional import NeuralFold


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
        pred, pred_s, _ = self.model(seq, reference=pair, verbose=True,
                                loss_pos_paired=self.loss_pos_paired, loss_neg_paired=self.loss_neg_paired, 
                                loss_pos_unpaired=self.loss_pos_unpaired, loss_neg_unpaired=self.loss_neg_unpaired)
        ref, ref_s, _ = self.model(seq, constraint=pair, max_internal_length=None, verbose=True)
        loss = pred - ref
        if self.verbose:
            print(seq)
            print(pred_s, pred.item())
            print(ref_s, ref.item())
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
    def __init__(self):
        self.train_loader = None
        self.test_loader = None


    def train(self, epoch):
        self.model.train()
        n_dataset = len(self.train_loader.dataset)
        loss_total, num = 0, 0
        running_loss, n_running_loss = 0, 0
        with tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, pairs in self.train_loader:
                n_batch = len(seqs)
                self.optimizer.zero_grad()
                loss = self.loss_fn(seqs, pairs, fname=fnames)
                loss_total += loss.item()
                num += n_batch
                loss.backward()
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
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss_total / num))


    def test(self, epoch):
        self.model.eval()
        n_dataset = len(self.test_loader.dataset)
        loss_total, num = 0, 0
        with torch.no_grad(), tqdm(total=n_dataset, disable=self.disable_progress_bar) as pbar:
            for fnames, seqs, pairs in self.test_loader:
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


    def run(self, args):
        self.disable_progress_bar = args.disable_progress_bar
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

        if args.model == 'Turner':
            self.model = RNAFold()
        elif args.model == 'NN':
            self.model = NeuralFold(args)
            if args.gpu >= 0:
                self.model.to(torch.device("cuda", args.gpu))
        else:
            raise('not implemented')

        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.l2_weight)
        elif args.optimizer =='AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, amsgrad=True, weight_decay=args.l2_weight)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
        elif args.optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), nesterov=True, lr=args.lr, momentum=0.9, weight_decay=args.l2_weight)
        elif args.optimizer == 'ASGD':
            self.optimizer = optim.ASGD(self.model.parameters(), lr=args.lr, weight_decay=args.l2_weight)
        else:
            raise('not implemented')

        self.loss_fn = StructuredLoss(self.model, 
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
            config = { '--model': args.model, '--param': args.param }
            if hasattr(self.model, "config"):
                config.update(self.model.config)
            with open(args.save_config, 'w') as f:
                for k, v in config.items():
                    if type(v) is bool: # pylint: disable=unidiomatic-typecheck
                        if v:
                            f.write('{}\n'.format(k))
                    elif isinstance(v, list) or isinstance(v, tuple):
                        for vv in v:
                            f.write('{}\n{}\n'.format(k, vv))
                    else:
                        f.write('{}\n{}\n'.format(k, v))


    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('train', help='training')
        # input
        subparser.add_argument('input', type=str,
                            help='Training data of BPSEQ-formatted file')
        subparser.add_argument('--test-input', type=str,
                            help='Test data of BPSEQ-formatted file')
        subparser.add_argument('--epochs', type=int, default=10, metavar='N',
                            help='number of epochs to train (default: 10)')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--param', type=str, default='param.pth',
                            help='output file name of trained parameters')
        subparser.add_argument('--model', choices=('Turner', 'NN'), default='Turner', 
                            help="Folding model ('Turner', 'NN')")
        subparser.add_argument('--log-dir', type=str, default=None,
                            help='Directory for storing logs')
        subparser.add_argument('--resume', type=str, default=None,
                            help='Checkpoint file for resume')
        subparser.add_argument('--save-config', type=str, default=None,
                            help='save model configurations')
        subparser.add_argument('--disable-progress-bar', action='store_true',
                            help='disable the progress bar in training')

        subparser.add_argument('--optimizer', choices=('Adam', 'AdamW', 'RMSprop', 'SGD', 'ASGD'), default='AdamW')
        subparser.add_argument('--l1-weight', type=float, default=0.,
                            help='the weight for L1 regularization (default: 0)')
        subparser.add_argument('--l2-weight', type=float, default=0.,
                            help='the weight for L2 regularization (default: 0)')
        subparser.add_argument('--lr', type=float, default=0.01,
                            help='the learning rate for optimizer (default: 0.01)')
        subparser.add_argument('--loss-pos-paired', type=float, default=0,
                            help='the penalty for positive base-pairs for loss augmentation (default: 0)')
        subparser.add_argument('--loss-neg-paired', type=float, default=0,
                            help='the penalty for negative base-pairs for loss augmentation (default: 0)')
        subparser.add_argument('--loss-pos-unpaired', type=float, default=0,
                            help='the penalty for positive unpaired bases for loss augmentation (default: 0)')
        subparser.add_argument('--loss-neg-unpaired', type=float, default=0,
                            help='the penalty for negative unpaired bases for loss augmentation (default: 0)')

        NeuralFold.add_args(subparser)

        subparser.set_defaults(func = lambda args: Train().run(args))
