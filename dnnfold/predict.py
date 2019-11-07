import argparse
import math
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .compbpseq import accuracy, compare_bpseq
from .dataset import BPseqDataset, FastaDataset
from .fold.nussinov import NussinovFold
from .fold.positional import NeuralFold
from .fold.rnafold import RNAFold


class Predict:
    def __init__(self):
        self.test_loader = None


    def predict(self, output_bpseq=None, result=None):
        res_fn = open(result, 'w') if result is not None else None
        self.model.eval()
        with torch.no_grad():
            for headers, seqs, _, refs in self.test_loader:
                start = time.time()
                rets = self.model.predict(seqs)
                elapsed_time = time.time() - start
                for header, seq, ref, (sc, pred, bp) in zip(headers, seqs, refs, rets):
                    if output_bpseq is None:
                        print('>'+header)
                        print(seq)
                        print(pred, "({:.1f})".format(sc))
                    elif output_bpseq == "stdout":
                        print('# {} (s={:.1f}, {:.5f}s)'.format(header, sc, elapsed_time))
                        for i in range(1, len(bp)):
                            print('{}\t{}\t{}'.format(i, seq[i-1], bp[i]))
                    else:
                        fn = os.path.basename(header)
                        fn = os.path.splitext(fn)[0] 
                        fn = os.path.join(output_bpseq, fn+".bpseq")
                        with open(fn, "w") as f:
                            f.write('# {} (s={:.1f}, {:.5f}s)\n'.format(header, sc, elapsed_time))
                            for i in range(1, len(bp)):
                                f.write('{}\t{}\t{}\n'.format(i, seq[i-1], bp[i]))
                    if res_fn is not None and len(ref) == len(bp):
                        x = compare_bpseq(ref, bp)
                        x = [header, len(seq), elapsed_time, sc] + list(x) + list(accuracy(*x))
                        res_fn.write(', '.join([str(v) for v in x]) + "\n")


    def run(self, args):
        try:
            test_dataset = FastaDataset(args.input)
        except RuntimeError:
            test_dataset = BPseqDataset(args.input)
        if len(test_dataset) == 0:
            test_dataset = BPseqDataset(args.input)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)

        if args.model == 'Turner':
            if args.param is not '':
                self.model = RNAFold()
            else:
                from . import param_turner2004
                self.model = RNAFold(param_turner2004)
        elif args.model == 'NN' or args.model == 'Zuker':
            self.model = NeuralFold(args)
        elif args.model == 'Nussinov':
            self.model = NussinovFold(args)
        else:
            raise('not implemented')

        if args.param is not '':
            p = torch.load(args.param)
            if isinstance(p, dict) and 'model_state_dict' in p:
                p = p['model_state_dict']
            self.model.load_state_dict(p)

        if args.gpu >= 0:
            self.model.to(torch.device("cuda", args.gpu))

        self.predict(output_bpseq=args.bpseq, result=args.result)


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
        subparser.add_argument('--model', choices=('Turner', 'NN', 'Zuker', 'Nussinov'), default='Turner', 
                            help="Folding model ('Turner', 'NN', 'Zuker', 'Nussinov')")
        subparser.add_argument('--param', type=str, default='',
                            help='file name of trained parameters') 
        subparser.add_argument('--result', type=str, default=None,
                            help='output the prediction accuracy if reference structures are given')
        subparser.add_argument('--bpseq', type=str, default=None,
                            help='output the prediction with BPSEQ format to the specified directory')

        NeuralFold.add_args(subparser)

        subparser.set_defaults(func = lambda args: Predict().run(args))
