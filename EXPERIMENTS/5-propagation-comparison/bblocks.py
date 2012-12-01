#!/usr/bin/python

import pylab
import sys
import numpy as np

import qopt
import qopt.algorithms
import qopt.problems

# f1
#schema = '01001**********'
#schema = '01*011*********'
#schema = '010*11*********'

# f2
#schema = '10100***************'
#schema = '1*1001**************'
#schema = '1010****************'

# f3
schema = '10001********************'
#schema = '1000*1*******************'
#schema = '1000*********************'

# sat 20
#schema = '***********1100*1***'
schema = '***********1*00*1***'
#schema = '*********001*00*****'


f = qopt.problems.func1d.f3
f = qopt.problems.sat20
chromlen = 20


def count_bblocks(pop):
    res = 0
    for chromo in pop:
        if qopt.matches(chromo, schema):
            res += 1
    return res

def evaluator(chrom):
    s = ''.join([str(g) for g in chrom])
    #x = f.getx(s)
    x = s
    return f.evaluate(x)

def step(ga):
    pop = []
    for ind in ga.getPopulation():
        chromo = ''.join([str(g) for g in ind])
        pop.append(chromo)
    bblocks_count.append(count_bblocks(pop))

Y = np.zeros((0,160))
for r in xrange(50):
    bblocks_count = []
    sga = qopt.algorithms.SGA(evaluator, chromlen)
    sga.setElitism(True)
    sga.setGenerations(160)
    sga.setPopulationSize(40)
    sga.stepCallback.set(step)
    sga.evolve()
    Y = np.vstack((Y, bblocks_count))

Y = np.average(Y, 0)

pylab.plot(Y, label='sga')

print Y

# QIGA

class QIGA(qopt.algorithms.QIGA):
    def generation(self):
        super(QIGA, self).generation()
        print self.P
        bblocks_count.append(count_bblocks(self.P))

qiga = QIGA(chromlen = chromlen, popsize = 10)
qiga.tmax = 160
qiga.problem = f

Y = np.zeros((0,160))
for r in xrange(50):
    bblocks_count = []
    qiga.run()
    print bblocks_count
    Y = np.vstack((Y, bblocks_count))

Y = np.average(Y, 0)
print Y

pylab.plot(Y, label='qiga')

pylab.legend(loc='lower right')
pylab.savefig('/tmp/result.pdf')

