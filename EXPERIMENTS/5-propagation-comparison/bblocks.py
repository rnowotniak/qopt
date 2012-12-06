#!/usr/bin/python
#encoding: utf-8

import pylab
import sys
import numpy as np

import qopt
import qopt.algorithms
import qopt.problems

MaxFE = 1600
repeat = 50
popsize = 10
sgapopsize = 80

prob = sys.argv[1]
chromlen = int(sys.argv[2])
schemanum = int(sys.argv[3])

fname = prob

if prob == 'f1':
    f = qopt.problems.func1d.f1
elif prob == 'f2':
    f = qopt.problems.func1d.f2
elif prob == 'f3':
    f = qopt.problems.func1d.f3
elif prob == 'sat':
    if chromlen == 15:
        f = qopt.problems.sat15
    elif chromlen == 20:
        f = qopt.problems.sat20
    elif chromlen == 25:
        f = qopt.problems.sat25
elif prob == 'k':
    if chromlen == 15:
        f = qopt.problems.knapsack15
    elif chromlen == 20:
        f = qopt.problems.knapsack20
    elif chromlen == 25:
        f = qopt.problems.knapsack25
else:
    assert False

fil = open(qopt.path('data/%s-%d-bblocks' % (prob, chromlen)))
schemata = [line.split()[0] for line in fil.readlines()]
schema = schemata[schemanum - 1]

#print schema
#sys.exit(0)

# f1
#schema = '01001**********'
#schema = '01*011*********'
#schema = '010*11*********'

# f2
#schema = '10100***************'
#schema = '1*1001**************'
#schema = '1010****************'

# f3
#schema = '10001********************'
#schema = '1000*1*******************'
#schema = '1000*********************'

# sat 15
#schema = '****001*11*****'
#schema = '**010*10*******'
#schema = '**01001********'

#schema = '***********1100*1***'
#schema = '***********1*00*1***'
#schema = '*********001*00*****'



def count_bblocks(pop):
    res = 0
    for chromo in pop:
        #print chromo
        #print schema
        if qopt.matches(chromo, schema):
            res += 1
        #print res
        #print '--'
    #print res
    return res

def evaluator(chrom):
    s = ''.join([str(g) for g in chrom])
    return f.evaluate(s) + 3

def step(ga):
    pop = []
    for ind in ga.getPopulation():
        chromo = ''.join([str(g) for g in ind])
        pop.append(chromo)
    bblocks_count.append(count_bblocks(pop))
    #print bblocks_count


# QIGA

class QIGA(qopt.algorithms.QIGA):
    def generation(self):
        super(QIGA, self).generation()
        #print self.Q[0]
        #e.append(qopt.E(self.Q, schema))
        #print len(e)
        bblocks_count.append(count_bblocks(self.P))

qiga = QIGA(chromlen = chromlen, popsize = popsize)
qiga.tmax = MaxFE / popsize
qiga.problem = f

Y = np.zeros((0,MaxFE / popsize))
for r in xrange(repeat):
    bblocks_count = []
    qiga.run()
    Y = np.vstack((Y, bblocks_count))

Y = np.average(Y, 0)
print Y

X = np.linspace(0,MaxFE,len(Y))
pylab.plot(X, Y / popsize, 'ro-', label='bQIGAo')

# SGA

Y = np.zeros((0,MaxFE / sgapopsize))
for r in xrange(repeat):
    bblocks_count = []
    sga = qopt.algorithms.SGA(evaluator, chromlen = chromlen, popsize = sgapopsize)
    sga.stepCallback.set(step)
    sga.setGenerations(MaxFE / sgapopsize)
    sga.evolve()
    Y = np.vstack((Y, bblocks_count))
Y = np.average(Y, 0)

pylab.plot(np.linspace(0,MaxFE,len(Y)), Y / sgapopsize, 's-', label='SGA')

pylab.title(u'Porównanie propagacji bloku budującego\nblok: %s, funkcja: %s, $chromlen=%d$' \
        % (schema, fname, chromlen))
pylab.ylabel(u'Chromosomy pasujące do bloku (\%)')
pylab.xlabel(u'Liczba wywołań funkcji oceny ($FE$)')
pylab.legend(loc='lower right')
pylab.savefig('/tmp/bb-%s-%d-%d.pdf' % (prob, chromlen, schemanum), bbox_inches = 'tight')

