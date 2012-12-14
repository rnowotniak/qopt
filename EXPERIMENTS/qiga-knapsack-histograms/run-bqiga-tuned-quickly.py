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

results = []

for r in xrange(20):
    bQIGAo.run()
    results.append(bQIGAo.bestval)

print numpy.average(results)
print numpy.var(results)

