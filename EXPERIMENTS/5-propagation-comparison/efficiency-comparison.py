#!/usr/bin/python

import sys
import numpy as np

import qopt.algorithms
import qopt.problems

#f = qopt.problems.func1d.f2
#f = qopt.problems.sat20
f = qopt.problems.knapsack20

def evaluator(chrom):
    s = ''.join([str(g) for g in chrom])
    s2 = '%s' % s
    f.repair(s2)
    #x = f.getx(s)
    x = s2
    return f.evaluate(x)

def step(ga):
    fvals.append(ga.bestIndividual().getRawScore())

Ysga = np.zeros((0,160))
for r in xrange(50):
    fvals = []
    sga = qopt.algorithms.SGA(evaluator, 20)
    sga.setElitism(True)
    sga.stepCallback.set(step)
    sga.setPopulationSize(40)
    sga.evolve()
    Ysga = np.vstack((Ysga, fvals))
Ysga = np.average(Ysga, 0)

print 'sga done'

class QIGA(qopt.algorithms.QIGA):
    def generation(self):
        super(QIGA, self).generation()
        #bbs = qopt.analysis.bblocks(self.P, '***101*1******')
        bbs = 5
        fvals.append(np.max(self.fvals))
        #print '%d %f %d' % (self.t, np.max(self.fvals), bbs)


qiga = QIGA(chromlen = 20, popsize = 40)
qiga.tmax = 160
qiga.problem = f

Y = np.zeros((0,160))
for r in xrange(50):
    fvals = []
    qiga.run()
    Y = np.vstack((Y, fvals))

Y = np.average(Y, 0)

print 'qiga done'

import pylab
X = range(160)
pylab.plot(X, Y, 'ro-', label='qiga')
pylab.plot(X, Ysga, 's-', label='sga')
pylab.legend(loc = 'lower right')
pylab.grid(True)
pylab.savefig('/tmp/result.pdf')

