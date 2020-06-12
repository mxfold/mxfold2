#!/usr/bin/env python

import sys

with open(sys.argv[1]) as f:
    print(">{}".format(sys.argv[1]))
    s = ''
    for l in f:
        l = l.rstrip().split(' ')
        s += l[1]
    print(s)