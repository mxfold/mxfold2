#%%
import os
from argparse import ArgumentParser
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import BPseqDataset
from .fold import RNAFold, CNNFold


class StructuredLoss(nn.Module):
    def __init__(self, model, pos_penalty, neg_penalty, l1_weight=0., l2_weight=0.):
        super(StructuredLoss, self).__init__()
        self.model = model
        self.pos_penalty = pos_penalty
        self.neg_penalty = neg_penalty
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight


    def forward(self, seq, pair, fname=None):
        pred = self.model(seq, reference=pair, pos_penalty=self.pos_penalty, neg_penalty=self.neg_penalty)
        ref = self.model(seq, constraint=pair, max_internal_length=len(seq))
        loss = pred - ref
        if loss.item()> 1e10 or torch.isnan(loss):
            print()
            print(fname)
            print(loss.item(), pred.item(), ref.item())
            print(seq)
            print(pair)

        if self.l1_weight > 0.0:
            for p in self.model.parameters():
                loss += self.l1_weight * torch.sum(torch.abs(p))

        if self.l2_weight > 0.0:
            l2_reg = 0.0
            for p in self.model.parameters():
                l2_reg += torch.sum(self.l2_weight * self.l2_weight * p * p)
            loss += torch.sqrt(l2_reg)

        return loss


class Train:
    def __init__(self):
        self.train_loader = None
        self.test_loader = None


    def train(self, epoch):
        self.model.train()
        loss_total, num = 0, 0
        with tqdm(total=len(self.train_loader.dataset)) as pbar:
            for fnames, seqs, pairs in self.train_loader:
                #seq, pair = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                loss = 0
                for fname, seq, pair in zip(fnames, seqs, pairs):
                    loss += self.loss_fn(seq, pair, fname=fname)
                    loss_total += loss.item()
                    num += 1
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix(train_loss='{:.3e}'.format(loss_total / num))
                pbar.update(len(seqs))
        print('Train Epoch: {}\tLoss: {:.6f}'.format(epoch, loss_total / num))


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
        train_dataset = BPseqDataset(args.input, unpaired='x')
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)

        if args.model == 'Turner':
            self.model = RNAFold()
        elif args.model == 'CNN':
            self.model = CNNFold(args)
        else:
            raise('never reach here')

        self.loss_fn = StructuredLoss(self.model, args.pos_penalty, args.neg_penalty, args.l1_weight, args.l2_weight)

        #self.optimizer = optim.SGD(self.model.parameters(), nesterov=True, lr=0.001, momentum=0.9)
        self.optimizer = optim.Adam(self.model.parameters())
        #self.optimizer = optim.Adagrad(self.model.parameters())

        checkpoint_epoch = 0
        if args.resume is not None:
            checkpoint_epoch = self.resume_checkpoint(args.resume)

        for epoch in range(checkpoint_epoch+1, args.epochs+1):
            self.train(epoch)
            # self.test()
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
        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--param', type=str, default='param.pth',
                            help='output file name of trained parameters')
        subparser.add_argument('--model', choices=('Turner', 'CNN'), default='Turner', 
                            help="Folding model ('Turner', 'CNN')")
        subparser.add_argument('--log-dir', type=str, default=None,
                            help='Directory for storing logs')
        subparser.add_argument('--resume', type=str, default=None,
                            help='Checkpoint file for resume')
        subparser.add_argument('--save-config', type=str, default=None,
                            help='save model configurations')

        subparser.add_argument('--l1-weight', type=float, default=0.,
                            help='the weight for L1 regularization (default: 0)')
        subparser.add_argument('--l2-weight', type=float, default=0.,
                            help='the weight for L2 regularization (default: 0)')
        subparser.add_argument('--pos-penalty', type=float, default=0,
                            help='the penalty for positive BPs for loss augmentation (default: 0)')
        subparser.add_argument('--neg-penalty', type=float, default=0,
                            help='the penalty for negative BPs for loss augmentation (default: 0)')

        CNNFold.add_args(subparser)

        subparser.set_defaults(func = lambda args: Train().run(args))
