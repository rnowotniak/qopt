#!/usr/bin/python

import pylab, numpy

import time
import sys,os
import qopt.algorithms
import qopt.problems

bQIGAo2 = qopt.algorithms.BQIGAo2(chromlen = 250)

#lut = bQIGAo.lookup_table.reshape((1,8))[0]
#.lut[3:8] = [.0, .044, .223, .254, .151]

bQIGAo2.mi = .99
print bQIGAo2.mi

bQIGAo2.tmax = 500
bQIGAo2.problem = qopt.problems.knapsack250

results = []

t1 = time.time()
for r in xrange(3000):
    bQIGAo2.run()
    print bQIGAo2.bestval
    sys.stdout.flush()
    # results.append(bQIGAo2.bestval)
t2 = time.time()

print (t2 - t1) / 3000

sys.exit(0)

print numpy.average(results)
print numpy.var(results)

