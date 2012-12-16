#!/usr/bin/python

import pylab, numpy

import time
import sys,os
import qopt.algorithms
import qopt.problems

alg = qopt.algorithms.BQIGAo2(chromlen = 718)

#alg.mi = .9918
alg.mi = .9807

# .992

# lut = alg.lookup_table.reshape((1,8))[0]
# lut[3:8] = [.0, .044, .223, .254, .151]

alg.tmax = 400
alg.problem = qopt.problems.sat718

results = []

for r in xrange(10):
    alg.run()
    results.append(alg.bestval)

print numpy.average(results)
print numpy.var(results)

f = open('/tmp/bqigao2-metaopt', 'a')
f.write('%g %g %g\n' % (alg.mi, numpy.average(results), numpy.var(results)))
f.close()

