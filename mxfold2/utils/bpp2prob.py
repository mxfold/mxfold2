import numpy as np
import sys

bpp = np.loadtxt(sys.argv[2])
ubpp = 1-np.sum(bpp, axis=0)

with open(sys.argv[1]) as f:
    for l in f:
        if not l.startswith('#'):
            i, b, j = l.rstrip().split()
            i, j = int(i),  int(j)
            if j>0:
                print(f'{bpp[i][j]:.02f};', end='')
            else:
                print(f'{ubpp[i]:.02f};', end='')
