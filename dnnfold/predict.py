import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import FastaDataset
from .fold import RNAFold


class Predict:
    def __init__(self):
        self.test_loader = None


    def predict(self):
        self.model.eval()
        with torch.no_grad():
            for headers, seqs in self.test_loader:
                # data = data.to(self.device)
                for header, seq in zip(headers, seqs):
                    sc, p = self.model.predict(seq)
                    # output = output.cpu().numpy()
                    print('>'+header)
                    print(seq)
                    print(p, "({:.1f})".format(sc))


    def run(self, args):
        test_dataset = FastaDataset(args.input)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # use_cuda = not args.no_cuda and torch.cuda.is_available()
        # self.device = torch.device("cuda" if use_cuda else "cpu") # pylint: disable=no-member
        # torch.manual_seed(args.seed)

        if args.model is not '':
            self.model = RNAFold()
            self.model.load_state_dict(torch.load(args.model))
        else:
            import dnnfold.default_param
            self.model = RNAFold(dnnfold.default_param)

        # self.model.to(self.device)
        self.predict()


    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('predict', help='predict')
        # input
        subparser.add_argument('input', type=str,
                            help='FASTA-formatted file')

        subparser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        subparser.add_argument('--model', type=str, default='',
                            help='file name of trained model') 

        subparser.set_defaults(func = lambda args: Predict().run(args))
