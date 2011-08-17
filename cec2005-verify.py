#!/usr/bin/python

import sys
import numpy as np
import cec2005

eps = 10e-5

# for i in xrange(10):
#     m = np.matrix(f[i])
#     print '% .15e' % cec2005.for_tests(m)

# sequential test
for fnum in xrange(1,4):
    print 'fnum: ', fnum
    fpt = open('input_data/test_data_func%d.txt' % fnum).readlines()
    i = 0
    xargs = []
    vals = []
    # read x_args and correct function values
    while True:
        m = np.matrix(fpt[0], np.double)
        if m.size == 1:
            break
        fpt.pop(0)
        xargs.append(m)
    for i in xrange(len(xargs)):
        m = np.double(fpt[0])
        fpt.pop(0)
        vals.append(m)

    f = getattr(cec2005, 'f%d' % fnum)
    for i in xrange(len(xargs)):
        calculated = f(xargs[i])
        print calculated, vals[i]
        assert abs(calculated - vals[i]) < eps

# parallel

# test execution time

#
# sequeantial
# for f in 1 .. 25:
#    good_resulsts = input_data/...
#    for line in input_data/...
#       if cec2005.f_n(line) == ...:
#           BAD
#
#
# parallel
#
#
# test time:
#
#
#

