from __future__ import annotations

import os
import random
import time
from argparse import Namespace
from pathlib import Path
from typing import Optional

import torch
#import torch.nn as nn
#import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader

from . import interface
from .compbpseq import accuracy, compare_bpseq
from .dataset import BPseqDataset, FastaDataset
from .fold.fold import AbstractFold
from .common import Common


class Predict(Common):
    def __init__(self):
        super(Predict, self).__init__()


    def predict(self, 
                model: AbstractFold | AveragedModel,
                data_loader: DataLoader,
                output_bpseq: Optional[str] = None, 
                output_bpp: Optional[str] = None, 
                result: Optional[str] = None, 
                use_constraint: bool = False,
                shape_list: Optional[list[str]] = None,
                shape_intercept: float = 0.0,
                shape_slope: float = 0.0) -> None:

        res_fn = open(result, 'w') if result is not None else None
        shape_list = [None] * len(data_loader) if shape_list is None else shape_list
        while len(shape_list) < len(data_loader):
            shape_list.append(None)

        model.eval()
        seq_processed = 0
        with torch.no_grad():
            for headers, seqs, vals in data_loader:
                start = time.time()
                if use_constraint:
                    constraint = [ tgt if tp=='BPSEQ' else None for tp, tgt in zip(vals['type'], vals['target'])]
                else:
                    constraint = None
                pseudoenergy = [ 
                    self.load_shape_reactivity(shape_file, shape_intercept, shape_slope) \
                        if shape_file is not None else None \
                        for shape_file in shape_list[seq_processed:seq_processed+len(seqs)] ]
                seq_processed += len(seqs)
                if output_bpp is None:
                    scs, preds, bps = model(seqs, constraint=constraint, pseudoenergy=pseudoenergy)
                    pfs = bpps = [None] * len(preds)
                else:
                    scs, preds, bps, pfs, bpps = model(seqs, return_partfunc=True, constraint=constraint, pseudoenergy=pseudoenergy)
                elapsed_time = time.time() - start
                for header, seq, ref, sc, pred, bp, pf, bpp in zip(headers, seqs, vals['target'], scs, preds, bps, pfs, bpps):
                    if output_bpseq is None:
                        print('>'+header)
                        print(seq)
                        print(pred, f'({sc:.1f})')
                    elif output_bpseq == "stdout":
                        print(f'# {header} (s={sc:.1f}, {elapsed_time:.5f}s)')
                        for i in range(1, len(bp)):
                            print(f'{i}\t{seq[i-1]}\t{bp[i]}')
                    else:
                        fn = os.path.basename(header)
                        fn = os.path.splitext(fn)[0] 
                        fn = os.path.join(output_bpseq, fn+".bpseq")
                        with open(fn, "w") as f:
                            print(f'# {header} (s={sc:.1f}, {elapsed_time:.5f}s)', file=f)
                            for i in range(1, len(bp)):
                                print(f'{i}\t{seq[i-1]}\t{bp[i]}', file=f)
                    if res_fn is not None:
                        x = compare_bpseq(ref, bp)
                        x = [header, len(seq), elapsed_time, sc.item()] + list(x) + list(accuracy(*x))
                        res_fn.write(', '.join([str(v) for v in x]) + "\n")
                    if output_bpp is not None:
                        fn = os.path.basename(header)
                        fn = os.path.splitext(fn)[0] 
                        fn = os.path.join(output_bpp, fn+".bpp")
                        with open(fn, "w") as f:
                            for i in range(1, len(bpp)):
                                print(f"{i} {seq[i-1]} ", end='', file=f)
                                for j, p in bpp[i]:
                                    print(f"{j}:{p:.3f}", end=' ', file=f)
                                print(file=f)


    def run(self, args: Namespace, conf: Optional[str] = None) -> None:
        torch.set_num_threads(args.threads)
        interface.set_num_threads(args.threads)

        test_dataset = FastaDataset(args.input)
        if len(test_dataset) == 0:
            test_dataset = BPseqDataset(args.input)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)

        model, _ = self.build_model(args)
        if args.param != '':
            param = Path(args.param)
            if not param.exists() and conf is not None:
                param = Path(conf).parent / param
            p = torch.load(param, map_location='cpu')
            if isinstance(p, dict) and 'model_state_dict' in p:
                p = p['model_state_dict']
            if 'n_averaged' in p:
                model = AveragedModel(model)
            model.load_state_dict(p)

        if args.gpu >= 0:
            model.to(torch.device("cuda", args.gpu))

        shape_list = None
        if args.shape is not None: 
            shape_list = []
            with open(args.shape) as f:
                for l in f:
                    l = l.rstrip('\n').split()
                    shape_list.append(l[0])
        elif args.shape_file is not None:
            shape_list = [args.shape_file]

        self.predict(model=model, data_loader=test_loader, 
                    output_bpseq=args.bpseq, output_bpp=args.bpp,
                    result=args.result, use_constraint=args.use_constraint,
                    shape_list=shape_list,
                    shape_intercept=args.shape_intercept, shape_slope=args.shape_slope)


    def load_shape_reactivity(self, fname: str, intercept: float = -0.8, slope: float = 2.6) -> torch.tensor:
        r = []
        with open(fname) as f:
            for l in f:
                l = l.rstrip('\n').split()
                if len(l) == 2:
                    idx, val = l
                elif len(l) == 3:
                    idx, _, val = l
                else:
                    raise(ValueError(f"Invalid SHAPE reactivity file: {fname}"))
                idx, val = int(idx), float(val)
                while len(r) < idx:
                    r.append(-999)
                r[idx-1] = val
        # Deigan’s pseudoenergy approach
        r = torch.tensor(r, dtype=float)
        not_na = r > -1
        r[torch.logical_not(not_na)] = 0
        r[not_na] = slope * torch.log(r[not_na]+1) + intercept
        return r

    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('predict', help='predict')
        # input
        subparser.add_argument('input', type=str,
                            help='FASTA-formatted file or list of BPseq files')
        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--threads', type=int, default=1, metavar='N',
                            help='the number of threads (default: 1)')
        subparser.add_argument('--param', type=str, default='',
                            help='file name of trained parameters') 
        subparser.add_argument('--use-constraint', default=False, action='store_true')
        subparser.add_argument('--result', type=str, default=None,
                            help='output the prediction accuracy if reference structures are given')
        subparser.add_argument('--bpseq', type=str, default=None,
                            help='output the prediction with BPSEQ format to the specified directory')
        subparser.add_argument('--bpp', type=str, default=None,
                            help='output the base-pairing probability matrix to the specified directory')
        subparser.add_argument('--shape', type=str, default=None, help='specify the file name that includes the list of SHAPE reactivity files')
        subparser.add_argument('--shape-file', type=str, default=None, help='specify the file name that includes SHAPE reactivity')
        subparser.add_argument('--shape-intercept', type=float, default=-0.8,
                            help='Specify an intercept used with SHAPE restraints. Default is -0.8 kcal/mol.')
        subparser.add_argument('--shape-slope', type=float, default=2.6, 
                            help='Specify a slope used with SHAPE restraints. Default is 2.6.')

        cls.add_fold_args(subparser)
        cls.add_network_args(subparser)

        subparser.set_defaults(func = lambda args, conf: Predict().run(args, conf))
