from __future__ import annotations

from argparse import Namespace
from .fold.fold import AbstractFold
from typing import Any

class Common:
    def init(self):
        pass

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

            elif args.fold == 'LinearFold':
                from .fold.linearfoldv import LinearFoldV
                if args.param == 'default' or args.param == 'turner2004':
                    args.param = ''
                    from . import param_turner2004
                    return LinearFoldV(init_param=param_turner2004), {}
                else:
                    return LinearFoldV(), {}

            elif args.fold == 'LinFold':
                from .fold.linfoldv import LinFoldV
                if args.param == 'default' or args.param == 'turner2004':
                    args.param = ''
                    from . import param_turner2004
                    return LinFoldV(init_param=param_turner2004), {}
                else:
                    return LinFoldV(), {}
        
        elif args.model == 'CONTRAfold':
            if args.fold == 'Zuker':
                from .fold.contrafold import CONTRAfold
                if args.param == 'default':
                    args.param = ''
                    from . import param_contrafold202
                    return CONTRAfold(init_param=param_contrafold202), {}
                else:
                    return CONTRAfold(), {}

            elif args.fold == 'LinearFold':
                from .fold.linearfoldc import LinearFoldC
                if args.param == 'default':
                    args.param = ''
                    from . import param_contrafold202
                    return LinearFoldC(param_contrafold202), {}
                else:
                    return LinearFoldC(), {}

            elif args.fold == 'LinFold':
                from .fold.linfoldc import LinFoldC
                if args.param == 'default':
                    args.param = ''
                    from . import param_contrafold202
                    return LinFoldC(param_contrafold202), {}
                else:
                    return LinFoldC(), {}

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
            #'bl_size': args.bl_size,
            'paired_opt': args.paired_opt,
            'mix_type': args.mix_type,
            'additional_params': args.additional_params,
        }

        model = None
        if args.model == 'Positional':
            if args.fold == 'Zuker':
                from .fold.zuker import ZukerFold
                model = ZukerFold(**config)

            elif args.fold == 'LinearFold':
                from .fold.linearfold import LinearFold
                model = LinearFold(**config)

            elif args.fold == 'LinFold':
                from .fold.linfold import LinFold
                model = LinFold(**config)

        elif args.model == 'Mix':
            from . import param_turner2004
            if args.fold == 'Zuker':
                from .fold.mix import MixedFold
                model = MixedFold(init_param=param_turner2004, **config)

            elif args.fold == 'LinearFold':
                from .fold.mix_linearfold import MixedLinearFold
                model = MixedLinearFold(init_param=param_turner2004, **config)

            elif args.fold == 'LinFold':
                from .fold.mix_linfold import MixedLinFold
                model = MixedLinFold(init_param=param_turner2004, **config)

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

            elif args.fold == 'LinearFold':
                from .fold.mix_linearfold1d import MixedLinearFold1D
                model = MixedLinearFold1D(init_param=param_turner2004, **config)

        # elif args.model == 'BL':
        #     if args.fold == 'Zuker':
        #         from .fold.zuker_bl import ZukerFoldBL
        #         model = ZukerFoldBL(**config)

        #     elif args.fold == 'LinearFold':
        #         from .fold.linearfold_bl import LinearFoldBL
        #         model = LinearFoldBL(**config)

        # elif args.model == 'MixBL':
        #     from . import param_turner2004
        #     if args.fold == 'Zuker':
        #         from .fold.mix_bl import MixedFoldBL
        #         model = MixedFoldBL(init_param=param_turner2004, **config)

        #     elif args.fold == 'LinearFold':
        #         from .fold.mix_linearfold_bl import MixedLinearFoldBL
        #         model = MixedLinearFoldBL(init_param=param_turner2004, **config)

        if model is None:
            raise(RuntimeError(f'not implemented: model={args.model}, fold={args.fold}'))

        return model, config

    @classmethod
    def add_network_args(cls, subparser):
        gparser = subparser.add_argument_group("Network setting")
        gparser.add_argument('--model', choices=('Turner', 'CONTRAfold', 'Positional', 'Mix', 'Mix1D', 'CFMix'), default='Turner', 
                        help="select parameter model (default: 'Turner')")
        gparser.add_argument('--additional-params', default=None, action='store_true')
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
        # gparser.add_argument('--bl-size', type=int, default=4,
        #                 help='the input dimension of the bilinear layer of LinearFold model (default: 4)')
        gparser.add_argument('--paired-opt', choices=('0_1_1', 'fixed', 'symmetric'), default='symmetric')
        gparser.add_argument('--mix-type', choices=('add', 'average'), default='average')
