from __future__ import annotations

from argparse import Namespace
from mxfold2.fold.fold import AbstractFold
from typing import Any
import torch

class Common:
    def init(self):
        pass

    @staticmethod
    def get_device(gpu: int) -> tuple[torch.device, str]:
        """Get appropriate device and device type for training/inference.

        Args:
            gpu: GPU ID (-1 for auto-detect, >=0 for specific CUDA GPU)

        Returns:
            tuple of (device, device_type):
                - device: torch.device to use
                - device_type: str ('cuda', 'mps', or 'cpu') for autocast/GradScaler

        Device selection logic:
            - gpu >= 0: Use specified CUDA GPU
            - gpu == -1: Auto-detect (MPS if available, else CPU)
        """
        if gpu >= 0:
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA GPU {gpu} requested but CUDA is not available")
            return torch.device("cuda", gpu), "cuda"
        else:
            # Auto-detect best available device
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                return torch.device("mps"), "mps"
            else:
                return torch.device("cpu"), "cpu"

    def build_model(self, args: Namespace) -> tuple[AbstractFold, dict[str, Any]]:
        if args.model == 'Turner':
            if args.fold == 'Zuker':
                from .fold.rnafold import RNAFold
                if args.param == 'default' or args.param == 'turner2004':
                    args.param = ''
                    from . import param_turner2004
                    return RNAFold(init_param=param_turner2004), {}
                else:
                    return RNAFold(), {}

            elif args.fold == 'LinFold' or args.fold == 'LinearFold':
                from .fold.linfoldv import LinFoldV
                if args.param == 'default' or args.param == 'turner2004':
                    args.param = ''
                    from . import param_turner2004
                    return LinFoldV(init_param=param_turner2004, beam_size=args.beam_size), {}
                else:
                    return LinFoldV(beam_size=args.beam_size), {}
        
        elif args.model == 'CONTRAfold':
            if args.fold == 'Zuker':
                from .fold.contrafold import CONTRAfold
                if args.param == 'default':
                    args.param = ''
                    from . import param_contrafold202
                    return CONTRAfold(init_param=param_contrafold202), {}
                else:
                    return CONTRAfold(), {}

            elif args.fold == 'LinFold' or args.fold == 'LinearFold':
                from .fold.linfoldc import LinFoldC
                if args.param == 'default':
                    args.param = ''
                    from . import param_contrafold202
                    return LinFoldC(param_contrafold202, beam_size=args.beam_size), {}
                else:
                    return LinFoldC(beam_size=args.beam_size), {}

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
            'dropout_rate_1d_cnn': args.dropout_rate_1d_cnn if args.dropout_rate_1d_cnn is not None else args.dropout_rate,
            'dropout_rate_2d_cnn': args.dropout_rate_2d_cnn if args.dropout_rate_2d_cnn is not None else args.dropout_rate,
            'dropout_rate_lstm': args.dropout_rate_lstm if args.dropout_rate_lstm is not None else args.dropout_rate,
            'fc_dropout_rate': args.fc_dropout_rate,
            'stochastic_depth_1d': args.stochastic_depth_1d,
            'stochastic_depth_2d': args.stochastic_depth_2d,
            'num_att': args.num_att,
            'pair_join': args.pair_join,
            'no_split_lr': args.no_split_lr,
            #'bl_size': args.bl_size,
            'paired_opt': args.paired_opt,
            'mix_type': args.mix_type,
            'weight_turner': args.weight_turner,
            'weight_positional': args.weight_positional,
            'additional_params': args.additional_params,
            'resnet_every_n': args.resnet_every_n,
        }

        model = None
        if args.model == 'Positional':
            if args.fold == 'Zuker':
                from .fold.zuker import ZukerFold
                model = ZukerFold(**config)

            elif args.fold == 'LinFold' or args.fold == 'LinearFold':
                from .fold.linfold import LinFold
                model = LinFold(beam_size=args.beam_size, **config)

        elif args.model == 'Mix':
            from . import param_turner2004
            if args.fold == 'Zuker':
                from .fold.mix import MixedFold
                model = MixedFold(init_param=param_turner2004, **config)

            elif args.fold == 'LinFold' or args.fold == 'LinearFold':
                from .fold.mix_linfold import MixedLinFold
                model = MixedLinFold(init_param=param_turner2004, beam_size=args.beam_size, **config)

        elif args.model == 'CFMix':
            from . import param_contrafold202
            if args.fold == 'Zuker':
                from .fold.cf_mix import CONTRAMixedFold
                model = CONTRAMixedFold(init_param=param_contrafold202, **config)

        # elif args.model == 'CFMixT':
        #     if args.fold == 'Zuker':
        #         from .fold.cf_mix import CONTRAMixedFold
        #         model = CONTRAMixedFold(tune_cf=True, **config)

        elif args.model == 'Mix1D':
            from . import param_turner2004
            if args.fold == 'Zuker':
                from .fold.mix1d import MixedFold1D
                model = MixedFold1D(init_param=param_turner2004, **config)

            # elif args.fold == 'LinearFold':
            #     from .fold.mix_linearfold1d import MixedLinearFold1D
            #     model = MixedLinearFold1D(init_param=param_turner2004, beam_size=args.beam_size, **config)

        # elif args.model == 'BL':
        #     if args.fold == 'Zuker':
        #         from .fold.zuker_bl import ZukerFoldBL
        #         model = ZukerFoldBL(**config)

        #     elif args.fold == 'LinearFold':
        #         from .fold.linearfold_bl import LinearFoldBL
        #         model = LinearFoldBL(beam_size=args.beam_size, **config)

        # elif args.model == 'MixBL':
        #     from . import param_turner2004
        #     if args.fold == 'Zuker':
        #         from .fold.mix_bl import MixedFoldBL
        #         model = MixedFoldBL(init_param=param_turner2004, **config)

        #     elif args.fold == 'LinearFold':
        #         from .fold.mix_linearfold_bl import MixedLinearFoldBL
        #         model = MixedLinearFoldBL(init_param=param_turner2004, beam_size=args.beam_size, **config)

        if model is None:
            raise(RuntimeError(f'not implemented: model={args.model}, fold={args.fold}'))

        return model, config

    @classmethod
    def add_fold_args(cls, subparser):
        gparser = subparser.add_argument_group("Folding setting")
        gparser.add_argument('--fold', choices=('Zuker', 'LinearFold', 'LinFold'), default='Zuker',
                            help="select folding algorithm (default: 'Zuker')")
        gparser.add_argument('--max-helix-length', type=int, default=30, 
                        help='the maximum length of helices (default: 30)')
        gparser.add_argument('--beam-size', type=int, default=100,
                        help='the beam size of LinFold algorithm (default: 100)')
                        
    @classmethod
    def add_network_args(cls, subparser):
        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--model', choices=('Turner', 'CONTRAfold', 'Positional', 'Mix', 'Mix1D', 'CFMix'), default='Mix', 
                        help="select parameter model (default: 'Mix')")
        gparser.add_argument('--additional-params', default=None, action='store_true')
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
        gparser.add_argument('--dropout-rate-1d-cnn', type=float, default=None,
                        help='dropout rate for 1D CNN layers (default: use --dropout-rate)')
        gparser.add_argument('--dropout-rate-2d-cnn', type=float, default=None,
                        help='dropout rate for 2D CNN layers (default: use --dropout-rate)')
        gparser.add_argument('--dropout-rate-lstm', type=float, default=None,
                        help='dropout rate for LSTM layers (default: use --dropout-rate)')
        gparser.add_argument('--fc-dropout-rate', type=float, default=0.0,
                        help='dropout rate of the hidden units (default: 0.0)')
        gparser.add_argument('--stochastic-depth-1d', type=str, default=None,
                        help='Survival probability for 1D CNN stochastic depth (comma-separated or single value, default: disabled)')
        gparser.add_argument('--stochastic-depth-2d', type=str, default=None,
                        help='Survival probability for 2D CNN stochastic depth (comma-separated or single value, default: disabled)')
        gparser.add_argument('--num-att', type=int, default=0,
                        help='the number of the heads of attention (default: 0)')
        gparser.add_argument('--pair-join', choices=('cat', 'add', 'mul', 'bilinear'), default='cat', 
                            help="how pairs of vectors are joined ('cat', 'add', 'mul', 'bilinear') (default: 'cat')")
        gparser.add_argument('--no-split-lr', default=False, action='store_true')
        # gparser.add_argument('--bl-size', type=int, default=4,
        #                 help='the input dimension of the bilinear layer of LinearFold model (default: 4)')
        gparser.add_argument('--paired-opt', choices=('0_1_1', 'fixed', 'symmetric'), default='symmetric')
        gparser.add_argument('--mix-type', choices=('add', 'average'), default='average')
        gparser.add_argument('--weight-turner', type=float, default=None,
                        help='weight for Turner/CONTRAfold model (overrides --mix-type)')
        gparser.add_argument('--weight-positional', type=float, default=None,
                        help='weight for positional model (overrides --mix-type)')
        gparser.add_argument('--weight-schedule', choices=('none', 'linear', 'cosine'), default='none',
                        help='weight scheduling strategy for mixed models (default: none)')
        gparser.add_argument('--weight-schedule-start', type=int, default=1,
                        help='epoch to start weight scheduling (default: 1)')
        gparser.add_argument('--weight-schedule-end', type=int, default=None,
                        help='epoch to end weight scheduling (default: total epochs)')
        gparser.add_argument('--resnet-every-n', type=int, default=1,
                        help='apply skip connection every N layers (default: 1 = every layer, 2 = classic ResNet)')
