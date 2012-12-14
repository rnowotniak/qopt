#!/usr/bin/python

import pylab, numpy

import sys,os
import qopt.algorithms
import qopt.problems

bQIGAo = qopt.algorithms.QIGA(chromlen = 250)
#bQIGAo = qopt.algorithms.QIGA_StorePriorToRepair(chromlen = 250)

bQIGAo.tmax = 500
bQIGAo.problem = qopt.problems.knapsack250

lut = bQIGAo.lookup_table.reshape((1,8))[0]
lut[3:8] = [.0, .044, .223, .254, .151]

#print bQIGAo.lookup_table.reshape((1,8))

#sys.exit(0)

if True:
    # this loop generates results-3000.txt
    for r in xrange(3000):
        bQIGAo.run()
        print bQIGAo.bestval
        sys.stdout.flush()
    sys.exit(0)

