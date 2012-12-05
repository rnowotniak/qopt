#!/usr/bin/python
#encoding: utf-8

import sys
import numpy as np

import qopt.algorithms
import qopt.problems

prob = sys.argv[1]

MaxFE = 1600
repeat = 50
popsize = 10
sgapopsize = 80
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
    return f.evaluate(s) + 3

def step(ga):
    fvals.append(ga.bestIndividual().getRawScore())

Ysga = np.zeros((0,MaxFE / sgapopsize))
for r in xrange(repeat):
    fvals = []
    sga = qopt.algorithms.SGA(evaluator, chromlen = chromlen, popsize = sgapopsize)
    sga.stepCallback.set(step)
    sga.setGenerations(MaxFE / sgapopsize)
    sga.evolve()
    Ysga = np.vstack((Ysga, fvals))
#ff = open('/tmp/bla','w')
#ff.write(str(Ysga))
#ff.close()
Ysga = np.average(Ysga, 0)
print 'sga done'

class QIGA(qopt.algorithms.QIGA):
    def generation(self):
        super(QIGA, self).generation()
        fvals.append(np.max(self.fvals))

qiga = QIGA(chromlen = chromlen, popsize = popsize)
qiga.tmax = MaxFE / popsize
qiga.problem = f

Y = np.zeros((0,MaxFE / popsize))
for r in xrange(repeat):
    fvals = []
    qiga.run()
    Y = np.vstack((Y, fvals))

Y = np.average(Y, 0)

print 'qiga done'

import pylab
X = np.linspace(0,MaxFE,len(Y))
pylab.plot(X, Y, 'ro-', label='bQIGAo')
pylab.plot(np.linspace(0,MaxFE,len(Ysga)), Ysga - 3, 's-', label='SGA')
pylab.legend(loc = 'lower right')
pylab.title(u'Porównanie efektywności algorytmów,\nfunkcja: $\\texttt{%s}$, $chromlen=%d$' % (prob, chromlen))
pylab.ylabel(u'Średnia wartość maksymalnego dopasowania')
pylab.xlabel(u'Liczba wywołań funkcji oceny ($FE$)')
pylab.grid(True)
pylab.savefig('/tmp/cmp-%s.pdf' % (prob), bbox_inches = 'tight')

