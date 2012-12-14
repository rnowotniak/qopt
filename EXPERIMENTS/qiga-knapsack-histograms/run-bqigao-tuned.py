#!/usr/bin/python

import pylab, numpy

import time
import sys,os
import qopt.algorithms
import qopt.problems

bQIGAo = qopt.algorithms.BQIGAo(chromlen = 250)

lut = bQIGAo.lookup_table.reshape((1,8))[0]
lut[3:8] = [.0, .044, .223, .254, .151]

bQIGAo.tmax = 500
bQIGAo.problem = qopt.problems.knapsack250

if True:
    # this loop generates results-3000.txt
    t1 = time.time()
    for r in xrange(3000):
        bQIGAo.run()
        print bQIGAo.bestval
        sys.stdout.flush()
    t2 = time.time()
    #print t2-t1
    sys.exit(0)

