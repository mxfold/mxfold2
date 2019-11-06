import argparse
import os
import random
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import FastaDataset, BPseqDataset
from .fold.rnafold import RNAFold
from .fold.positional import NeuralFold
from .fold.nussinov import NussinovFold


class Predict:
    def __init__(self):
        self.test_loader = None


    def predict(self, output_bpseq=None, result='xxx'):
        res_fn = open(result, 'w') if result is not None else None
        self.model.eval()
        with torch.no_grad():
            for headers, seqs, _, refs in self.test_loader:
                print(refs)
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
                    if res_fn is not None:
                        x = self.compare_bpseq(ref, pred)
                        x = [header, len(seq), elapsed_time, sc] + list(x) + list(self.accuracy(*x))
                        res_fn.write(', '.join([str(v) for v in x]) + "\n")


    def compare_bpseq(self, ref, pred):
        print(ref, pred)
        assert(len(ref) == len(pred))
        tp = fp = fn = 0
        for i, (j1, j2) in enumerate(zip(ref, pred)):
            if j1 > 0 and i < j1: # pos
                if j1 == j2:
                    tp += 1
                elif j2 > 0 and i < j2:
                    fp += 1
                    fn += 1
                else:
                    fn += 1
            elif j2 > 0 and i < j2:
                fp += 1
        tn = len(ref) * (len(ref) - 1) // 2 - tp - fp - fn
        return (tp, tn, fp, fn)


    def accuracy(self, tp, tn, fp, fn):
        sen = tp / (tp + fn) if tp+fn > 0. else 0.
        ppv = tp / (tp + fp) if tp+fp > 0. else 0.
        fval = 2 * sen * ppv / (sen + ppv) if sen+ppv > 0. else 0.
        mcc = ((tp*tn)-(fp*fn)) / math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) > 0. else 0.
        return (sen, ppv, fval, mcc)



    def run(self, args):
        try:
            test_dataset = FastaDataset(args.input)
        except RuntimeError:
            test_dataset = BPseqDataset(args.input)
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
        self.predict(output_bpseq=args.bpseq)


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
