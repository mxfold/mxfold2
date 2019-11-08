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


    def build_model(self, args):
        if args.model == 'Turner':
            if args.param is not '':
                return RNAFold(), {}
            else:
                from . import param_turner2004
                return RNAFold(param_turner2004), {}

        config = {
            'num_filters': args.num_filters if args.num_filters is not None else (256,),
            'motif_len': args.motif_len if args.motif_len is not None else (7,),
            'pool_size': args.pool_size if args.pool_size is not None else (1,),
            'dilation': args.dilation, 
            'num_lstm_layers': args.num_lstm_layers, 
            'num_lstm_units': args.num_lstm_units,
            'num_hidden_units': args.num_hidden_units if args.num_hidden_units is not None else (128,),
            'dropout_rate': args.dropout_rate,
            'use_bilinear': args.use_bilinear,
            'lstm_cnn': args.lstm_cnn,
            'context_length': args.context_length                
        }

        if args.model == 'NN' or args.model == 'Zuker':
            model = NeuralFold(**config)

        elif args.model == 'Nussinov':
            model = NussinovFold(**config)

        else:
            raise('not implemented')

        return model, config


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

        self.model, _ = self.build_model(args)
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
                            help='FASTA-formatted file or list of BPseq files')

        subparser.add_argument('--seed', type=int, default=0, metavar='S',
                            help='random seed (default: 0)')
        subparser.add_argument('--gpu', type=int, default=-1, 
                            help='use GPU with the specified ID (default: -1 = CPU)')
        subparser.add_argument('--param', type=str, default='',
                            help='file name of trained parameters') 
        subparser.add_argument('--result', type=str, default=None,
                            help='output the prediction accuracy if reference structures are given')
        subparser.add_argument('--bpseq', type=str, default=None,
                            help='output the prediction with BPSEQ format to the specified directory')

        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--model', choices=('Turner', 'NN', 'Zuker', 'Nussinov'), default='Turner', 
                            help="Folding model ('Turner', 'NN', 'Zuker', 'Nussinov')")
        gparser.add_argument('--num-filters', type=int, action='append',
                        help='the number of CNN filters')
        gparser.add_argument('--motif-len', type=int, action='append',
                        help='the length of each filter of CNN')
        gparser.add_argument('--pool-size', type=int, action='append',
                        help='the width of the max-pooling layer of CNN')
        gparser.add_argument('--dilation', type=int, default=0, 
                        help='Use the dilated convolution')
        gparser.add_argument('--num-lstm-layers', type=int, default=0,
                        help='the number of the LSTM hidden layers')
        gparser.add_argument('--num-lstm-units', type=int, default=0,
                        help='the number of the LSTM hidden units')
        gparser.add_argument('--num-hidden-units', type=int, action='append',
                        help='the number of the hidden units of full connected layers')
        gparser.add_argument('--dropout-rate', type=float, default=0.0,
                        help='dropout rate of the hidden units')
        gparser.add_argument('--use-bilinear', default=False, action='store_true')
        gparser.add_argument('--lstm-cnn', default=False, action='store_true',
                        help='use LSTM layer before CNN')
        gparser.add_argument('--context-length', type=int, default=1,
                        help='the length of context for FC layers')

        subparser.set_defaults(func = lambda args: Predict().run(args))
