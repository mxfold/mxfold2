#!/usr/bin/env python

import sys

with open(sys.argv[1]) as f:
    print('>'+sys.argv[1])
    for l in f:
        idx, c, pair = l.rstrip('\n').split()
        print(c, end='')
    print()