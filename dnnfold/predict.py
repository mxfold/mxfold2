import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import FastaDataset
from .fold import RNAFold, CNNFold


class Predict:
    def __init__(self):
        self.test_loader = None


    def predict(self, use_bpseq):
        self.model.eval()
        with torch.no_grad():
            for headers, seqs in self.test_loader:
                # data = data.to(self.device)
                for header, seq in zip(headers, seqs):
                    sc, pred, bp = self.model.predict(seq)
                    # output = output.cpu().numpy()
                    if use_bpseq:
                        print('# {} ({:.1f})'.format(header, sc))
                        for i in range(1, len(bp)):
                            print('{}\t{}\t{}'.format(i, seq[i-1], bp[i]))
                    else:
                        print('>'+header)
                        print(seq)
                        print(pred, "({:.1f})".format(sc))


    def run(self, args):
        test_dataset = FastaDataset(args.input)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # use_cuda = not args.no_cuda and torch.cuda.is_available()
        # self.device = torch.device("cuda" if use_cuda else "cpu") # pylint: disable=no-member
        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)

        if args.model == 'Turner':
            if args.param is not '':
                self.model = RNAFold()
                self.model.load_state_dict(torch.load(args.param))
            else:
                from . import param_turner2004
                self.model = RNAFold(param_turner2004)
        elif args.model == 'CNN':
            self.model = CNNFold(args)
            if args.param is not '':
                self.model.load_state_dict(torch.load(args.param))
        else:
            raise('never reach here')

        # self.model.to(self.device)
        self.predict(args.bpseq)


    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('predict', help='predict')
        # input
        subparser.add_argument('input', type=str,
                            help='FASTA-formatted file')

        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--model', choices=('Turner', 'CNN'), default='Turner', 
                            help="Folding model ('Turner', 'CNN')")
        subparser.add_argument('--param', type=str, default='',
                            help='file name of trained parameters') 
        subparser.add_argument('--bpseq', action='store_true',
                            help='output the prediction with BPSEQ format')

        CNNFold.add_args(subparser)

        subparser.set_defaults(func = lambda args: Predict().run(args))
