#!/usr/bin/python
# encoding=utf-8

# http://www.scipy.org/Cookbook/Matplotlib/AdjustingImageSize

import sys
import qopt, qopt.problems
import pylab

#pylab.rc('text', usetex=True)
#pylab.rc('text.latex', unicode=True)

import qopt.problems._sat
s1 = qopt.problems._sat.SatProblem(qopt.path('problems/sat/flat30-100.cnf'))
s2 = qopt.problems._sat.SatProblem(qopt.path('problems/sat/qg4-08.cnf'))
s3 = qopt.problems._sat.SatProblem(qopt.path('problems/sat/hanoi4.cnf'))

k10 = qopt.problems.knapsack10
k100 = qopt.problems.knapsack100
k250 = qopt.problems.knapsack250
k500 = qopt.problems.knapsack500

def plot(prob, length, fname):
    Y = []
    for i in xrange(2**14):
        kstr = qopt.int2bin(i, 14) + '0' * (length - 14)
        Y.append(prob.evaluate(kstr))
    pylab.figure()
    pylab.plot(Y)
    pylab.xlim([0,2**14])
    pylab.grid(True)
    pylab.savefig(fname, bbox_inches = 'tight')

plot(s1, 90, '/tmp/sat1.pdf')
plot(s2, 512, '/tmp/sat2.pdf')
plot(s3, 718, '/tmp/sat3.pdf')

plot(k100, 100, '/tmp/knapsack100.pdf')
plot(k250, 250, '/tmp/knapsack250.pdf')
plot(k500, 500, '/tmp/knapsack500.pdf')


f1 = qopt.problems.func1d.f1
f2 = qopt.problems.func1d.f2
f3 = qopt.problems.func1d.f3

pylab.figure()

X = pylab.linspace(0, 200, 200)
pylab.plot(
        [0, 10, 20, 30, 40, 56, 60, 65, 80, 90, 100, 120, 150, 180, 200],
        [0, 20, 40, 10, 25, 33, 80, 45, 60, 20, 0, 20, 40, 20, 0],
        'ro', markersize = 10, label=u'Węzły interpolacji')
pylab.plot(X, [f1.evaluate(x) for x in X], label=u'Funkcja interpolująca $f_1$')
#pylab.xlim((-5,5))
pylab.legend(loc='upper right')
pylab.grid(True)
pylab.xlabel('$x$')
pylab.ylabel('$f_1(x)$')
pylab.gcf().get_size_inches()[0] *= 2
pylab.savefig('/tmp/f1.pdf', bbox_inches = 'tight')
pylab.cla()

X = pylab.linspace(-5, 5, 200)
pylab.plot(X, [f2.evaluate(x) for x in X], label='Funkcja $f_2(x)$')
pylab.xlim((-5,5))
pylab.grid(True)
pylab.legend(loc='upper left')
pylab.xlabel('$x$')
pylab.ylabel('$f_2(x)$')
#pylab.gca().set_aspect(1, 'box')
pylab.savefig('/tmp/f2.pdf', bbox_inches = 'tight')
pylab.cla()

X = pylab.linspace(0, 17, 200)
pylab.plot(
        [0, 1, 2, 2.75, 3.4, 4.2, 5, 6, 6.6, 7.2, 8, 9, 9.8, 10.5, 11.2, 13, 14.5, 16.5],
        [1.6, 2.3, 2.4, 2.5, -1, 2, 3.3, 3.75, 1.1, 2.2, 4.6, 4.8, 5, .7, 3, 1.5, 4, 3],
        'ro', markersize = 10, label=u'Węzły interpolacji')
pylab.plot(X, [f3.evaluate(x) for x in X], label=u'Funkcja interpolująca $f_3$')
pylab.xlim((0,17))
pylab.grid(True)
pylab.legend(loc='lower right')
pylab.xlabel('$x$')
pylab.ylabel('$f_3(x)$')
#pylab.gca().set_aspect(1, 'box')
pylab.savefig('/tmp/f3.pdf', bbox_inches = 'tight')
