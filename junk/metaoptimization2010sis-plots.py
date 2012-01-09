#!/usr/bin/python

import sys
from copy import deepcopy

import qopt
import qopt.algorithms.iQIEA as iQIEA
import qopt.analysis.plot as plot

iqiea = iQIEA.iQIEA()

iqiea.fitness_function = iQIEA.testfuncs_f1
iqiea.G = 30
iqiea.bounds = [(-30,30)] * iqiea.G
iqiea.popsize = 10
iqiea.K = 10
iqiea.kstep = 10
iqiea.XI = 0.1
iqiea.DELTA = 0.5
iqiea.maxiter = 200

qopt.tic()

algs = []
for x in xrange(5):
    iqiea.initialize()
    iqiea.run()
    algs.append(deepcopy(iqiea))

plot1 = plot.Plot(algs)
plot1.save('/tmp/avg.png')

print '%f seconds' % qopt.toc()

