#!/usr/bin/python

import pylab, numpy
import time
import sys,os
import qopt.algorithms
import qopt.problems

bQIGAo = qopt.algorithms.BQIGAo2(chromlen = 250)
#bQIGAo = qopt.algorithms.QIGA_StorePriorToRepair(chromlen = 250)

bQIGAo.tmax = 500
bQIGAo.problem = qopt.problems.knapsack250

if True:
    # this loop generates results-3000.txt
    t1 = time.time()
    for r in xrange(1000):
        bQIGAo.run()
        print bQIGAo.bestval
        sys.stdout.flush()
    t2 = time.time()
    print (t2 - t1) / 1000
    sys.exit(0)

