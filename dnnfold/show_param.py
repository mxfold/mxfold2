from argparse import ArgumentParser
import torch
from .fold import RNAFold

class ShowParam:
    def __init__(self):
        pass

    def run(self, args):
        model = RNAFold()
        if args.checkpoint:
            checkpoint = torch.load(args.model)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(torch.load(args.model))
        print(model.state_dict())

    @classmethod
    def add_args(cls, parser):
        subparser = parser.add_parser('show_param', help='show parameters')
        # input
        subparser.add_argument('model', type=str,
                            help='model file to show parameter')
        subparser.add_argument('--checkpoint', action='store_true', 
                            help='show parameters in checkpoint file')
        subparser.set_defaults(func = lambda args: ShowParam().run(args))
