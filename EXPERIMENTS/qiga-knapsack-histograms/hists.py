#!/usr/bin/python

import pylab, numpy

import sys,os
import qopt.algorithms
import qopt.problems

k100 = numpy.matrix(';'.join(open('results-knapsack100.txt').readlines()))
k250 = numpy.matrix(';'.join(open('results-3000.txt').readlines()))
k250_2 = numpy.matrix(';'.join(open('results-3000-cap0.3.txt').readlines()))
k500 = numpy.matrix(';'.join(open('results-knapsack500.txt').readlines()))

pylab.figure()

pylab.subplot(4,1,1)
pylab.hist(k100, 480)
pylab.grid(True)

pylab.subplot(4,1,2)
pylab.hist(k250, 480)
pylab.xlim((1350, 1490))
pylab.grid(True)

pylab.subplot(4,1,3)
pylab.hist(k250_2, 480)
pylab.grid(True)

pylab.subplot(4,1,4)
pylab.hist(k500, 480)
pylab.grid(True)


pylab.savefig('/tmp/hists.pdf', bbox_inches = 'tight')

