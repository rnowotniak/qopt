#!/usr/bin/python

import pylab, numpy

import sys,os
import qopt.algorithms
import qopt.problems

bQIGAo = qopt.algorithms.QIGA(chromlen = 250)
bQIGAo.tmax = 500
bQIGAo.problem = qopt.problems.knapsack

if True:
    # this loop generates results-3000.txt
    for r in xrange(3000):
        bQIGAo.run()
        print bQIGAo.bestval
        sys.stdout.flush()
    sys.exit(0)

