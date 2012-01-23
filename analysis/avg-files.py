#!/usr/bin/python

import sys
import numpy as np

s = None
for fname in sys.argv[1:]:
    m = np.matrix(';'.join(open(fname, 'r').readlines()))
    if s == None:
        s = m
    else:
        s += m

s /= len(sys.argv[1:])
for p in s:
    print p[0,0], p[0,1]

