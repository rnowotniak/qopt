#!/usr/bin/python

import pylab, numpy

import sys,os
import qopt.algorithms
import qopt.problems

# k100 = numpy.matrix(';'.join(open('results-knapsack100.txt').readlines()))
# k250 = numpy.matrix(';'.join(open('results-3000.txt').readlines()))
# k250_2 = numpy.matrix(';'.join(open('results-3000-cap0.3.txt').readlines()))
# k500 = numpy.matrix(';'.join(open('results-knapsack500.txt').readlines()))

h1 = numpy.matrix(';'.join(open('results-3000.txt').readlines()))
h2 = numpy.matrix(';'.join(open('results-3000-tosamo.txt').readlines()))
h3 = numpy.matrix(';'.join(open('results-3000-StorePrior.txt').readlines()))

pylab.figure()

pylab.subplot(3,1,1)
pylab.hist(h1, 480)

pylab.subplot(3,1,2)
pylab.hist(h2, 480)

pylab.subplot(3,1,3)
pylab.hist(h3, 480)


pylab.savefig('/tmp/hists.pdf', bbox_inches = 'tight')
sys.exit(0)

pylab.subplot(4,1,1)
pylab.hist(k100, 480)
pylab.grid(True)

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

