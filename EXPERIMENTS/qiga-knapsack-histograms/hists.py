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
h2 = numpy.matrix(';'.join(open('results-3000-bqigao.txt').readlines()))
h3 = numpy.matrix(';'.join(open('results-3000-tuned.txt').readlines()))
h4 = numpy.matrix(';'.join(open('results-3000-bqigao-tuned.txt').readlines()))

print numpy.average(h1), numpy.var(h1)
print numpy.average(h2), numpy.var(h2)
print numpy.average(h3), numpy.var(h3)
print numpy.average(h4), numpy.var(h4)

pylab.subplot(4,1,1)
pylab.title('QIGA$(\\theta)$')
pylab.xlim((1380, 1500))
pylab.hist(h1, 300)

pylab.subplot(4,1,2)
pylab.title('bQIGAo')
pylab.xlim((1380, 1500))
pylab.hist(h2, 300)

pylab.subplot(4,1,3)
pylab.title('QIGA$(\\theta)$ tuned')
pylab.xlim((1380, 1500))
pylab.hist(h3, 300)

pylab.subplot(4,1,4)
pylab.title('bQIGAo tuned')
pylab.xlim((1380, 1500))
pylab.hist(h4, 300)

pylab.savefig('/tmp/hists.pdf', bbox_inches = 'tight')
sys.exit(0)

####

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

