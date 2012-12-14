#!/usr/bin/python
#encoding: utf-8

import sys
import numpy as np

import qopt.algorithms
import qopt.problems

#prob = sys.argv[1]

prob = 'knapsack250'

MaxFE = 4000
repeat = 50
popsize = 10
sgapopsize = 200
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
elif prob == 'knapsack100':
    f = qopt.problems.knapsack100
elif prob == 'knapsack250':
    f = qopt.problems.knapsack250
else:
    assert False

if prob in ('f1', 'sat15', 'knapsack15'):
    chromlen = 15
elif prob in ('f2', 'sat20', 'knapsack20'):
    chromlen = 20
elif prob in ('f3', 'sat25', 'knapsack25'):
    chromlen = 25
elif prob in ('knapsack100',):
    chromlen = 100
elif prob in ('knapsack250',):
    chromlen = 250
else:
    assert False



class QIGA2(qopt.algorithms.BQIGAo2):
    def generation(self):
        super(QIGA2, self).generation()
        fvals.append(np.max(self.fvals))

qiga2 = QIGA2(chromlen = chromlen, popsize = popsize)
qiga2.tmax = MaxFE / popsize
qiga2.problem = f
qiga2.mi = .9918

Yqiga2 = np.zeros((0,MaxFE / popsize))
for r in xrange(repeat):
    fvals = []
    qiga2.run()
    Yqiga2 = np.vstack((Yqiga2, fvals))

Yqiga2 = np.average(Yqiga2, 0)

print 'qiga2 done'



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


qiga = QIGA(chromlen = chromlen, popsize = popsize)
qiga.tmax = MaxFE / popsize
qiga.problem = f

lut = qiga.lookup_table.reshape((1,8))[0]
lut[3:8] = [.0, .044, .223, .254, .151]

Ytuned1 = np.zeros((0,MaxFE / popsize))
for r in xrange(repeat):
    fvals = []
    qiga.run()
    Ytuned1 = np.vstack((Ytuned1, fvals))

Ytuned1 = np.average(Ytuned1, 0)

print 'qiga1 tuned done'


def evaluator(chrom):
    s = ''.join([str(g) for g in chrom])
    return f.evaluate(s) + 3

def step(ga):
    fvals.append(ga.bestIndividual().getRawScore())

Ysga = np.zeros((0,MaxFE / sgapopsize))
for r in xrange(repeat):
    fvals = []
    sga = qopt.algorithms.SGA(evaluator, chromlen = chromlen, popsize = sgapopsize, elitism = True)
    sga.stepCallback.set(step)
    sga.setGenerations(MaxFE / sgapopsize)
    sga.evolve()
    Ysga = np.vstack((Ysga, fvals))
#ff = open('/tmp/bla','w')
#ff.write(str(Ysga))
#ff.close()
Ysga = np.average(Ysga, 0)
print 'sga done'

import pylab
X = np.linspace(0,MaxFE,len(Y))
pylab.plot(X, Yqiga2, 'ro-', label='bQIGAo2')
pylab.plot(X, Ytuned1, 'm^-', label='bQIGAo1-tuned')
pylab.plot(X, Y, 'gs-', label='bQIGAo1')
pylab.plot(np.linspace(0,MaxFE,len(Ysga)), Ysga - 3, 'x-', label='SGA')
pylab.legend(loc = 'lower right')
pylab.title(u'Porównanie efektywności algorytmów,\nfunkcja: $\\texttt{%s}$, $chromlen=%d$' % (prob, chromlen))
pylab.ylabel(u'Średnia wartość maksymalnego dopasowania')
pylab.xlabel(u'Liczba wywołań funkcji oceny ($FE$)')
pylab.grid(True)
pylab.savefig('/tmp/cmp-%s.pdf' % (prob), bbox_inches = 'tight')

