#!/usr/bin/python

import sys
from copy import deepcopy

import qopt
import qopt.algorithms.iQIEA as iQIEA
import qopt.analysis.plot as plot

iqiea = iQIEA.iQIEA()

iqiea.fitness_function = iQIEA.testfuncs_f2
iqiea.G = 30
iqiea.bounds = [(-10,10)] * iqiea.G
iqiea.popsize = 5
iqiea.K = 5
iqiea.kstep = 8
iqiea.XI = 0.1
iqiea.DELTA = 0.5
iqiea.maxiter = 250

qopt.tic()

algs = []
for x in xrange(30):
    iqiea.initialize()
    iqiea.run()
    algs.append(deepcopy(iqiea))

plot1 = plot.Plot(algs)
plot1.pylab().ylim(ymax=1000)
plot1.pylab().xlim(xmax=2500)
plot1.save('/tmp/avg.png')

print '%f seconds' % qopt.toc()


#
#
# plot.Compare(iqiea, rqiea) ?
#

