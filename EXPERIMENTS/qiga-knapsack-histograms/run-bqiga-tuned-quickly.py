#!/usr/bin/python

import pylab, numpy

import time
import sys,os
import qopt.algorithms
import qopt.problems

alg = qopt.algorithms.BQIGAo2(chromlen = 250)

alg.mi = .9918

# .992

# lut = alg.lookup_table.reshape((1,8))[0]
# lut[3:8] = [.0, .044, .223, .254, .151]

alg.tmax = 500
alg.problem = qopt.problems.knapsack250

results = []

for r in xrange(60):
    alg.run()
    results.append(alg.bestval)

print numpy.average(results)
print numpy.var(results)

