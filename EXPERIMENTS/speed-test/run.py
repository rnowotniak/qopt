#!/usr/bin/python

import sys
import time
import random

import qopt.problems

runs = 1000

def benchmarkCEC2013(fnum):
    f = qopt.problems.CEC2013(fnum)
    t0 = time.time()
    x = [random.randrange(-100,100) for i in xrange(10)]
    for i in xrange(runs):
        f.evaluate(x)
    t1 = time.time()
    return (t1 - t0) / runs

def benchmark(prob, s):
    t0 = time.time()
    for i in xrange(runs):
        prob.evaluate(s)
    t1 = time.time()
    return (t1 - t0) / runs

def benchmarkCEC2005(fnum):
    t0 = time.time()
    for i in xrange(runs):
        f = qopt.problems.cec2005(fnum)
        f.evaluate([0] * 10)
    t1 = time.time()
    return (t1 - t0) / runs

def benchmarkCassini():
    cassini = qopt.problems.cec2011(15)
    t0 = time.time()
    runs = 100
    for i in xrange(runs):
        res = cassini.evaluate([random.randrange(-1000,0), 3.265804135361, 0.528440291493, 0.382390419772,\
                        167.937610148996, 424.032204472497, 53.304869390732, 589.767895836123, 2199.961911685212,\
                                0.772877728290, 0.531757418755, 0.010789195916, 0.167388829033, 0.010425709182,\
                                        1.358596310427, 1.050001151443, 1.306852313623, 69.813404643644, -1.593310577644,\
                                                -1.959572311812, -1.554796022348, -1.513432303179])
    t1 = time.time()
    return (t1 - t0) / runs

def benchmarkMessenger():
    f = qopt.problems.cec2011(15)
    t0 = time.time()
    runs = 100
    for i in xrange(runs):
        x=[random.randrange(1900,2300),3,.5,.5,200,200,200,200,200,200,.2,.2,.2,.2,.2,.2,2,2,2,2,2,0,0,1,1]
        res = f.evaluateMessenger(x)
    t1 = time.time()
    return (t1 - t0) / runs

s = ''.join([random.choice(('0','1')) for i in xrange(100)])
print 'Knapsack 100',
print benchmark(qopt.problems.knapsack100, s)

s = ''.join([random.choice(('0','1')) for i in xrange(250)])
print 'Knapsack 250',
print benchmark(qopt.problems.knapsack250, s)

s = ''.join([random.choice(('0','1')) for i in xrange(500)])
print 'Knapsack 500',
print benchmark(qopt.problems.knapsack500, s)

s = ''.join([random.choice(('0','1')) for i in xrange(90)])
print 'SAT 90',
print benchmark(qopt.problems.sat90, s)

s = ''.join([random.choice(('0','1')) for i in xrange(512)])
print 'SAT 512',
print benchmark(qopt.problems.sat512, s)

s = ''.join([random.choice(('0','1')) for i in xrange(718)])
print 'SAT 718',
print benchmark(qopt.problems.sat718, s)

for fnum in xrange(1,26):
    res = benchmarkCEC2005(fnum)
    print 'CEC05 f%d: %g' % (fnum,res)

print 'Cassini: %g' % benchmarkCassini()
print 'Messenger: %g' % benchmarkMessenger()

for fnum in xrange(1,29):
    res = benchmarkCEC2013(fnum)
    print 'CEC13 f%d: %g' % (fnum, res)


