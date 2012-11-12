#!/usr/bin/python

import sys
import numpy as np

import qopt.algorithms
import qopt.problems


class QIGA(qopt.algorithms.QIGA):
    def generation(self):
        super(QIGA, self).generation()
        #bbs = qopt.analysis.bblocks(self.P, '***101*1******')
        bbs = 5
        fvals.append(np.max(self.fvals))
        #print '%d %f %d' % (self.t, np.max(self.fvals), bbs)


qiga = QIGA(chromlen = 250)
qiga.tmax = 160
qiga.problem = qopt.problems.knapsack

Y = np.zeros((0,160))
for r in xrange(25):
    fvals = []
    qiga.run()
    Y = np.vstack((Y, fvals))

Y = np.average(Y, 0)

#print qopt.problems.func1d.f1.getx(qiga.best[:20])
#print qiga.best[:20]

import pylab
X = range(160)
pylab.plot(X, Y, 'o-')
pylab.grid(True)
pylab.savefig('/tmp/result.pdf')

