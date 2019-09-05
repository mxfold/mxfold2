#%%
import dnnfold
from .param import Param
from .dataset import FastaDataset

def run(input_fasta):
    p = Param(dnnfold.default_param)
    a = FastaDataset(input_fasta)
    for h, s in a:
        sc, r = dnnfold.predict(s, p)
        print(">"+h)
        print(s)
        print(r+" ({:.1f})".format(sc))
        print()

#%%
import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DNNfold',
        fromfile_prefix_chars='@',
        add_help=True
    )
    parser.add_argument('input_fasta', type=str, help='FASTA-formatted file')
    parser.add_argument('--param', type=str, help='parameter file')
    args = parser.parse_args()
    run(args.input_fasta)
