#!/usr/bin/python

import sys
import numpy as np

import qopt.algorithms
import qopt.problems

def evaluator(chrom):
    s = ''.join([str(g) for g in chrom])
    f = qopt.problems.func1d.f1
    x = f.getx(s)
    return f.evaluate(x)

def step(ga):
    fvals.append(ga.bestIndividual().getRawScore())

Ysga = np.zeros((0,160))
for r in xrange(50):
    fvals = []
    sga = qopt.algorithms.SGA(evaluator, 20)
    sga.setElitism(True)
    sga.stepCallback.set(step)
    sga.evolve()
    Ysga = np.vstack((Ysga, fvals))
Ysga = np.average(Ysga, 0)


class QIGA(qopt.algorithms.QIGA):
    def generation(self):
        super(QIGA, self).generation()
        #bbs = qopt.analysis.bblocks(self.P, '***101*1******')
        bbs = 5
        fvals.append(np.max(self.fvals))
        #print '%d %f %d' % (self.t, np.max(self.fvals), bbs)


qiga = QIGA(chromlen = 20)
qiga.tmax = 160
qiga.problem = qopt.problems.func1d.f1

Y = np.zeros((0,160))
for r in xrange(25):
    fvals = []
    qiga.run()
    Y = np.vstack((Y, fvals))

Y = np.average(Y, 0)



import pylab
X = range(160)
pylab.plot(X, Y, 'ro-')
pylab.plot(X, Ysga, 's-')
pylab.grid(True)
pylab.savefig('/tmp/result.pdf')

