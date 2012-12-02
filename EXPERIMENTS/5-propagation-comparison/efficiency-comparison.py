#!/usr/bin/python
#encoding: utf-8

import sys
import numpy as np

import qopt.algorithms
import qopt.problems

prob = sys.argv[1]

repeat = 50
popsize = 10
#chromlen = 15

#f = qopt.problems.func1d.f3
#f = qopt.problems.sat25
#f = qopt.problems.knapsack15

if prob == 'f1':
    f = qopt.problems.func1d.f1
elif prob == 'f2':
    f = qopt.problems.func1d.f2
elif prob == 'f3':
    f = qopt.problems.func1d.f3
elif prob == 'sat15':
    f = qopt.problems.sat15
elif prob == 'sat20':
    f = qopt.problems.sat20
elif prob == 'sat25':
    f = qopt.problems.sat25
elif prob == 'knapsack15':
    f = qopt.problems.knapsack15
elif prob == 'knapsack20':
    f = qopt.problems.knapsack20
elif prob == 'knapsack25':
    f = qopt.problems.knapsack25

if prob in ('f1', 'sat15', 'knapsack15'):
    chromlen = 15
elif prob in ('f2', 'sat20', 'knapsack20'):
    chromlen = 20
elif prob in ('f3', 'sat25', 'knapsack25'):
    chromlen = 25

def evaluator(chrom):
    s = ''.join([str(g) for g in chrom])
    s = '%s' % s
    f.repair(s)
    #x = f.getx(s)
    x = s
    return f.evaluate(x)

def step(ga):
    fvals.append(ga.bestIndividual().getRawScore())

Ysga = np.zeros((0,160))
for r in xrange(repeat):
    fvals = []
    sga = qopt.algorithms.SGA(evaluator, chromlen)
    sga.setElitism(True)
    sga.stepCallback.set(step)
    sga.setPopulationSize(popsize)
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


qiga = QIGA(chromlen = chromlen, popsize = popsize)
qiga.tmax = 160
qiga.problem = f

Y = np.zeros((0,160))
for r in xrange(repeat):
    fvals = []
    qiga.run()
    Y = np.vstack((Y, fvals))

Y = np.average(Y, 0)

print 'qiga done'

import pylab
X = range(160)
pylab.plot(X, Y, 'ro-', label='QIGA')
pylab.plot(X, Ysga, 's-', label='SGA')
pylab.legend(loc = 'lower right')
pylab.title(u'Porównanie efektywności algorytmów,\nfunkcja: $\\texttt{%s}$, $chromlen=%d$' % (prob, chromlen))
pylab.ylabel(u'Średnia wartość dopasowania')
pylab.xlabel(u'Numer generacji $t$')
pylab.grid(True)
pylab.savefig('/tmp/cmp-%s.pdf' % (prob), bbox_inches = 'tight')

