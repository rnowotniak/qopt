#!/usr/bin/python

import sys
import qopt.algorithms.iQIEA as iQIEA
import qopt.analysis.plot as plot
import copy

iqiea = iQIEA.iQIEA()
iqiea.fitness_function = iQIEA.testfuncs_f3
iqiea.bounds = [(-600,600)]*30
iqiea.popsize = 10
iqiea.K = 10
iqiea.G = 30
iqiea.kstep = 5
iqiea.XI = .23
iqiea.DELTA = .785
iqiea.maxiter = 800

algs = []
for run in xrange(50):
    iqiea.run()
    algs.append(copy.deepcopy(iqiea))

plot1 = plot.Plot(algs)

import os
os.system('bash rcqiea-start.sh')

import numpy as np
mces = []
for n in xrange(1, 51):
    mces.append(plot.readDataFromFile('/tmp/iQIEA-accordance/%02d.log' % n))
data = np.mean(mces, 0)

import pylab

pylab.plot(data[:,0], data[:,1], '-', label = 'metaoptimization2010sis implementation')
pylab.xlim(0, 8000)
pylab.ylim(0, 500)
pylab.grid(True)
pylab.legend()
pylab.savefig('/tmp/consistency.pdf')


