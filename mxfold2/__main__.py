import os
import sys
from argparse import ArgumentParser

from .predict import Predict
from .train import Train
#from .show_param import ShowParam

default_conf = os.path.join(os.path.dirname(__file__), 'models', 'TrainSetAB.conf')

def main():
    parser = ArgumentParser(
        description='RNA secondary structure prediction using deep learning with thermodynamic integration',
        fromfile_prefix_chars='@',
        add_help=True)
    subparser = parser.add_subparsers(title='Sub-commands')
    parser.set_defaults(func = lambda args, conf: parser.print_help())
    Train.add_args(subparser)
    Predict.add_args(subparser)
    # ShowParam.add_args(subparser)
    args = parser.parse_args()

    if hasattr(args, 'param'):
        if args.param == '':
            sys.argv.append('@'+default_conf)
            args = parser.parse_args()
        elif args.param == 'turner2004':
            args.param = ''

    conf = list(filter(lambda x: x[0]=='@', sys.argv))
    conf = None if len(conf)==0 else conf[-1][1:]

    args.func(args, conf)

if __name__ == '__main__':
    main()
