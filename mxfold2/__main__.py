from argparse import ArgumentParser
import sys

from .train import Train
from .predict import Predict
from .show_param import ShowParam

def main(args=None):
    conf = list(filter(lambda x: x[0]=='@', sys.argv))
    conf = None if len(conf)==0 else conf[-1][1:]
    
    parser = ArgumentParser(
        description='RNA secondary structure prediction using deep learning with thermodynamic integrations',
        fromfile_prefix_chars='@',
        add_help=True)
    subparser = parser.add_subparsers(title='Sub-commands')
    parser.set_defaults(func = lambda args: parser.print_help())
    Train.add_args(subparser)
    Predict.add_args(subparser)
    # ShowParam.add_args(subparser)
    args = parser.parse_args(args=args)
    args.func(args, conf)

if __name__ == '__main__':
    main()
