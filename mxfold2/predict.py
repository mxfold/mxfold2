import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from .compbpseq import accuracy, compare_bpseq
from .dataset import BPseqDataset, FastaDataset
from .fold.mix import MixedFold
from .fold.rnafold import RNAFold
from .fold.zuker import ZukerFold


class Predict:
    def __init__(self):
        self.test_loader = None


    def predict(self, output_bpseq=None, output_bpp=None, result=None, use_constraint=False):
        res_fn = open(result, 'w') if result is not None else None
        self.model.eval()
        with torch.no_grad():
            for headers, seqs, refs in self.test_loader:
                start = time.time()
                if output_bpp is None:
                    if use_constraint:
                        scs, preds, bps = self.model(seqs, constraint=refs)
                    else:
                        scs, preds, bps = self.model(seqs)
                    pfs = bpps = [None] * len(preds)
                else:
                    if use_constraint:
                        scs, preds, bps, pfs, bpps = self.model(seqs, return_partfunc=True, constraint=refs)
                    else:
                        scs, preds, bps, pfs, bpps = self.model(seqs, return_partfunc=True, )
                elapsed_time = time.time() - start
                for header, seq, ref, sc, pred, bp, pf, bpp in zip(headers, seqs, refs, scs, preds, bps, pfs, bpps):
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
                        bpp = np.triu(bpp)
                        bpp = bpp + bpp.T
                        fn = os.path.basename(header)
                        fn = os.path.splitext(fn)[0] 
                        fn = os.path.join(output_bpp, fn+".bpp")
                        np.savetxt(fn, bpp, fmt='%.5f')


    def build_model(self, args):
        if args.model == 'Turner':
            if args.param != '':
                return RNAFold(), {}
            else:
                from . import param_turner2004
                return RNAFold(param_turner2004), {}

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
        }

        if args.model == 'Zuker':
            model = ZukerFold(model_type='M', **config)

        elif args.model == 'ZukerC':
            model = ZukerFold(model_type='C', **config)

        elif args.model == 'ZukerL':
            model = ZukerFold(model_type='L', **config)

        elif args.model == 'ZukerS':
            model = ZukerFold(model_type='S', **config)

        elif args.model == 'Mix':
            from . import param_turner2004
            model = MixedFold(init_param=param_turner2004, **config)

        elif args.model == 'MixC':
            from . import param_turner2004
            model = MixedFold(init_param=param_turner2004, model_type='C', **config)

        else:
            raise('not implemented')

        return model, config


    def run(self, args, conf=None):
        test_dataset = FastaDataset(args.input)
        if len(test_dataset) == 0:
            test_dataset = BPseqDataset(args.input)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        if args.seed >= 0:
            torch.manual_seed(args.seed)
            random.seed(args.seed)

        self.model, _ = self.build_model(args)
        if args.param != '':
            param = Path(args.param)
            if not param.exists() and conf is not None:
                param = Path(conf).parent / param
            p = torch.load(param, map_location='cpu')
            if isinstance(p, dict) and 'model_state_dict' in p:
                p = p['model_state_dict']
            self.model.load_state_dict(p)

        if args.gpu >= 0:
            self.model.to(torch.device("cuda", args.gpu))

        self.predict(output_bpseq=args.bpseq, output_bpp=args.bpp, result=args.result, use_constraint=args.use_constraint)


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
        subparser.add_argument('--param', type=str, default='',
                            help='file name of trained parameters') 
        subparser.add_argument('--use-constraint', default=False, action='store_true')
        subparser.add_argument('--result', type=str, default=None,
                            help='output the prediction accuracy if reference structures are given')
        subparser.add_argument('--bpseq', type=str, default=None,
                            help='output the prediction with BPSEQ format to the specified directory')
        subparser.add_argument('--bpp', type=str, default=None,
                            help='output the base-pairing probability matrix to the specified directory')

        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--model', choices=('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'ZukerC', 'Mix', 'MixC'), default='Turner', 
                            help="Folding model ('Turner', 'Zuker', 'ZukerS', 'ZukerL', 'ZukerC', 'Mix', 'MixC')")
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

        subparser.set_defaults(func = lambda args, conf: Predict().run(args, conf))
