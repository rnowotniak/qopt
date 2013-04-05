#!/usr/bin/python

import sys
import numpy as np

import qopt.problems
# from qopt.algorithms import MyRQIEA2
from qopt.algorithms import MyRQIEA2

REPEAT = 10

myrqiea2 = MyRQIEA2(chromlen = 10, popsize = 10)
# myrqiea2._initialize()
myrqiea2.tmax = 100

print '%2s %5s %12s %12s' % ('f', 'opt', 'mean', 'stddev')

for fnum in xrange(1,29):
    fobj = qopt.problems.CEC2013(fnum)
    optval = fobj.evaluate(fobj.optimum)
    myrqiea2.problem = fobj

    results = []
    for r in xrange(REPEAT):
        myrqiea2.run()
        results.append(myrqiea2.bestval)
    results = np.array(results)

    print '%2d %5g %12g %12g' % (fnum, optval, results.mean(), results.std())


