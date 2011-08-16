#!/usr/bin/python

import sys
import numpy as np
import cec2005

f = open('input_data/test_data_func1.txt').readlines()

for i in xrange(10):
    m = np.matrix(f[i])
    print '% .15e' % cec2005.for_tests(m)

