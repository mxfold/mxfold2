import sys
from argparse import ArgumentParser
from pathlib import Path

from mxfold2.predict import Predict
from mxfold2.train import Train

DEFAULT_CONFIG = Path(__file__).resolve().parent / 'models' / 'nr80_imle_rank.conf'


def main():
    parser = ArgumentParser(
        description='RNA secondary structure prediction using deep learning with thermodynamic integration',
        fromfile_prefix_chars='@',
        add_help=True)
    subparser = parser.add_subparsers(title='Sub-commands')
    parser.set_defaults(func=lambda args, conf: parser.print_help())
    Train.add_args(subparser)
    Predict.add_args(subparser)

    argv = sys.argv[1:]
    args = parser.parse_args(argv)
    conf = next((arg[1:] for arg in reversed(argv) if arg.startswith('@')), None)
    if argv[:1] == ['predict'] and args.param in ('', 'default'):
        if args.model in ('Turner', 'CONTRAfold'):
            args.param = 'default'
        else:
            conf = str(DEFAULT_CONFIG)
            args = parser.parse_args(['predict', f'@{conf}', *argv[1:]])
            args.param = str(DEFAULT_CONFIG.with_suffix('.pth'))

    args.func(args, conf)


if __name__ == '__main__':
    main()
