#!/usr/bin/python
#encoding: utf-8

import sys
import time
import numpy as np

import qopt.algorithms
import qopt.problems

#   # ais -- bardzo zle
#   # bejing -- niezle
#   # knapsack -- rewelacja
#   # ii (inductive)) -- ZLE
#   # jnh --  tuned bardzo dobrze, potem qiga2, potem reszta, bardzo stabilnie i czytelnie
#   # sw100-8-lp1-c5   -- orig, 2, tuned, sga  -- stabilnie
#   # sw100-8-lp6-c5   --  prawie identycznie j.w.
#   # CBSy --   tuned, 2, orig, sga ;  stabilnie, powtarzalnie
#   
#   # prob = sys.argv[1]
#   prob = 'mysat'
#   f = qopt.problems.SatProblem('/tmp/uf175-013.cnf')
#   chromlen = 175
#   timecache = np.array([0.,0.,0.,0.])
#   readCache = False
#   

MaxFE = 5000

#   repeat = 20
#   popsize = 10
#   sgapopsize = 80
#   #chromlen = 15
#   
#   #f = qopt.problems.func1d.f3
#   #f = qopt.problems.sat25
#   #f = qopt.problems.knapsack15
#   
#   if prob == 'f1':
#       f = qopt.problems.func1d.f1
#   elif prob == 'f2':
#       f = qopt.problems.func1d.f2
#   elif prob == 'f3':
#       f = qopt.problems.func1d.f3
#   elif prob == 'sat15':
#       f = qopt.problems.sat15
#   elif prob == 'sat20':
#       f = qopt.problems.sat20
#   elif prob == 'sat25':
#       f = qopt.problems.sat25
#   elif prob == 'sat90':
#       f = qopt.problems.sat90
#   elif prob == 'sat512':
#       f = qopt.problems.sat512
#   elif prob == 'sat718':
#       f = qopt.problems.sat718
#   elif prob == 'knapsack15':
#       f = qopt.problems.knapsack15
#   elif prob == 'knapsack20':
#       f = qopt.problems.knapsack20
#   elif prob == 'knapsack25':
#       f = qopt.problems.knapsack25
#   elif prob == 'knapsack100':
#       f = qopt.problems.knapsack100
#   elif prob == 'knapsack250':
#       f = qopt.problems.knapsack250
#   elif prob == 'knapsack400':
#       f = qopt.problems.knapsack400
#   elif prob == 'knapsack500':
#       f = qopt.problems.knapsack500
#   
#   if prob in ('sat15', 'knapsack15'):
#       chromlen = 15
#   elif prob in ('sat20', 'knapsack20'):
#       chromlen = 20
#   elif prob in ('sat25', 'knapsack25'):
#       chromlen = 25
#   elif prob in ('sat90',):
#       chromlen = 90
#   elif prob in ('sat512',):
#       chromlen = 512
#   elif prob in ('sat718',):
#       chromlen = 718
#   elif prob in ('f1', 'knapsack100',):
#       chromlen = 100
#   elif prob in ('f2', 'knapsack250',):
#       chromlen = 250
#   elif prob in ('f3', 'knapsack500',):
#       chromlen = 500
#   elif prob in ('knapsack400',):
#       chromlen = 400
#   
#   def evaluator(chrom):
#       s = ''.join([str(g) for g in chrom])
#       return f.evaluate(s) + 3
#   
#   def step(ga):
#       fvals.append(ga.bestIndividual().getRawScore())
#   
#   if not readCache:
#       Ysga = np.zeros((0,MaxFE / sgapopsize))
#       t0 = time.time()
#       for r in xrange(repeat):
#           fvals = []
#           sga = qopt.algorithms.SGA(evaluator, chromlen = chromlen, popsize = sgapopsize, elitism = True)
#           sga.stepCallback.set(step)
#           sga.setGenerations(MaxFE / sgapopsize)
#           sga.evolve()
#           Ysga = np.vstack((Ysga, fvals))
#       t1 = time.time()
#       timecache[0] = (t1-t0)
#       print 'sga time: %f' % (t1-t0)
#       Ysga = np.average(Ysga, 0)
#       np.save('/tmp/cache-sga', Ysga)
#   else:
#       Ysga = np.load('/tmp/cache-sga.npy')
#   print 'sga done'
#   
#   class QIGA(qopt.algorithms.QIGA):
#       def generation(self):
#           super(QIGA, self).generation()
#           fvals.append(np.max(self.fvals))
#   
#   qiga = QIGA(chromlen = chromlen, popsize = popsize)
#   qiga.tmax = MaxFE / popsize
#   qiga.problem = f
#   
#   if not readCache:
#       Y = np.zeros((0,MaxFE / popsize))
#       t0 = time.time()
#       for r in xrange(repeat):
#           fvals = []
#           qiga.run()
#           Y = np.vstack((Y, fvals))
#       t1 = time.time()
#       timecache[1] = (t1-t0)
#       print 'bQIGAo1 time: %f' % (t1-t0)
#       Y = np.average(Y, 0)
#       np.save('/tmp/cache-bQIGAo1', Y)
#   else:
#       Y = np.load('/tmp/cache-bQIGAo1.npy')
#   print 'qiga done'
#   
#   
#   qiga = QIGA(chromlen = chromlen, popsize = popsize)
#   qiga.tmax = MaxFE / popsize
#   qiga.problem = f
#   
#   lut = qiga.lookup_table.reshape((1,8))[0]
#   lut[3:8] = [.0, .044, .223, .254, .151]
#   
#   if not readCache:
#       Ytuned1 = np.zeros((0,MaxFE / popsize))
#       t0 = time.time()
#       for r in xrange(repeat):
#           fvals = []
#           qiga.run()
#           Ytuned1 = np.vstack((Ytuned1, fvals))
#       t1 = time.time()
#       timecache[2] = (t1-t0)
#       print 'bQIGAo1-tuned time: %f' % (t1-t0)
#       Ytuned1 = np.average(Ytuned1, 0)
#       np.save('/tmp/cache-bQIGAo1-tuned', Ytuned1)
#   else:
#       Ytuned1 = np.load('/tmp/cache-bQIGAo1-tuned.npy')
#   print 'qiga1 tuned done'
#   
#   class QIGA2(qopt.algorithms.BQIGAo2):
#       def generation(self):
#           super(QIGA2, self).generation()
#           fvals.append(np.max(self.fvals))
#   
#   qiga2 = QIGA2(chromlen = chromlen, popsize = popsize)
#   qiga2.tmax = MaxFE / popsize
#   qiga2.problem = f
#   qiga2.mi = .9918 # bardzo dobre dla knapsacka
#   
#   if not readCache:
#       Yqiga2 = np.zeros((0,MaxFE / popsize))
#       t0 = time.time()
#       for r in xrange(repeat):
#           fvals = []
#           qiga2.run()
#           Yqiga2 = np.vstack((Yqiga2, fvals))
#       t1 = time.time()
#       timecache[3] = (t1-t0)
#       print 'bQIGAo2 time: %f' % (t1-t0)
#       Yqiga2 = np.average(Yqiga2, 0)
#       np.save('/tmp/cache-bQIGAo2', Yqiga2)
#   else:
#       Yqiga2 = np.load('/tmp/cache-bQIGAo2.npy')
#   print 'qiga2 done'
#   
#   print timecache
#   
#   np.save('/tmp/cache-times', timecache)

import glob
import pylab

prob = 'bejing-252'

def genplot(prob):
    pylab.cla()
    Yqiga2 = np.load('/var/tmp/cache/%s/cache-bQIGAo2.npy' % prob)
    Ysga = np.load('/var/tmp/cache/%s/cache-sga.npy' % prob)
    Y = np.load('/var/tmp/cache/%s/cache-bQIGAo1.npy' % prob)
    Ytuned1 = np.load('/var/tmp/cache/%s/cache-bQIGAo1-tuned.npy' % prob)


    cnffilename = glob.glob('/var/tmp/cache/%s/*.cnf' % prob)[0]

    line = filter(lambda l: l.startswith('p '), open(cnffilename, 'r').readlines())[0]
    chromlen = int(line.split(' ')[2])

    print prob,
    print ';',
    print chromlen,
    print ';',
    print Ysga[-1], ';',
    print Y[-1], ';',
    print Ytuned1[-1], ';',
    print Yqiga2[-1], ';'

    X = np.linspace(0,MaxFE,len(Y))
    pylab.ylim((1300,1480))
    #pylab.plot(X, Yqiga2, 'ro-', label='QIGA-2')
    pylab.plot(X, Ytuned1, 'm^-', label='QIGA-1 tuned')
    pylab.plot(X, Y, 'gs-', label='QIGA-1')
    pylab.plot(np.linspace(0,MaxFE,len(Ysga)), Ysga - 3, 'x-', label='SGA')
    pylab.legend(loc = 'lower right')
    pylab.title(u'Porównanie efektywności algorytmów,\nZadanie: $\\texttt{%s}$, rozmiar $N=%d$' % (prob, chromlen))
    pylab.ylabel(u'Średnia wartość maksymalnego dopasowania')
    pylab.xlabel(u'Liczba wywołań funkcji oceny ($FE$)')
    pylab.grid(True)

    # pylab.savefig('/tmp/cmp-%s.pdf' % (prob), bbox_inches = 'tight')
    pylab.savefig('/tmp/wykres-%s.pdf' % prob, bbox_inches = 'tight')

for prob in glob.glob('cache/???*'):
    pass
    #genplot(prob.split('/')[-1])

genplot('knapsack250')

