import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import FastaDataset
from .fold.rnafold import RNAFold
from .fold.positional import NeuralFold
from .fold.nussinov import NussinovFold


class Predict:
    def __init__(self):
        self.test_loader = None


    def predict(self, use_bpseq=None):
        self.model.eval()
        with torch.no_grad():
            for headers, seqs in self.test_loader:
                start = time.time()
                rets = self.model.predict(seqs)
                elapsed_time = time.time() - start
                for header, seq, (sc, pred, bp) in zip(headers, seqs, rets):
                    if use_bpseq is None:
                        print('>'+header)
                        print(seq)
                        print(pred, "({:.1f})".format(sc))
                    elif use_bpseq == "stdout":
                        print('# {} (s={:.1f}, {:.5f}s)'.format(header, sc, elapsed_time))
                        for i in range(1, len(bp)):
                            print('{}\t{}\t{}'.format(i, seq[i-1], bp[i]))
                    else:
                        fn = os.path.basename(header)
                        fn = os.path.splitext(fn)[0] 
                        fn = os.path.join(use_bpseq, fn+".bpseq")
                        with open(fn, "w") as f:
                            f.write('# {} (s={:.1f}, {:.5f}s)\n'.format(header, sc, elapsed_time))
                            for i in range(1, len(bp)):
                                f.write('{}\t{}\t{}\n'.format(i, seq[i-1], bp[i]))


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
        elif args.model == 'NN':
            self.model = NeuralFold(args)
            if args.param is not '':
                self.model.load_state_dict(torch.load(args.param))
            if args.gpu >= 0:
                self.model.to(torch.device("cuda", args.gpu))
        elif args.model == 'Nussinov':
            self.model = NussinovFold(args)
            if args.param is not '':
                self.model.load_state_dict(torch.load(args.param))
            if args.gpu >= 0:
                self.model.to(torch.device("cuda", args.gpu))
        else:
            raise('never reach here')

        # self.model.to(self.device)
        self.predict(use_bpseq=args.bpseq)


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
        subparser.add_argument('--model', choices=('Turner', 'NN', 'Nussinov'), default='Turner', 
                            help="Folding model ('Turner', 'NN', 'Nussinov')")
        subparser.add_argument('--param', type=str, default='',
                            help='file name of trained parameters') 
        subparser.add_argument('--bpseq', type=str, default=None,
                            help='output the prediction with BPSEQ format to the specified directory')

        NeuralFold.add_args(subparser)

        subparser.set_defaults(func = lambda args: Predict().run(args))
