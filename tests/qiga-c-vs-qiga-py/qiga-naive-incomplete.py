#!/usr/bin/python

#
# This naive and INCOMPLETE implementation is already 50x slower than implementation in C
#

import sys
import time

import numpy as np

chromlen = 250
popsize = 10
maxgen = 500

Q = np.zeros((popsize, chromlen))
P = [list('0' * chromlen) for x in xrange(popsize)]
fvals = np.zeros(popsize)
best = np.zeros(chromlen)
bestval = 0

def initialize():
    for i in xrange(popsize):
        for j in xrange(chromlen):
            Q[i][j] = np.pi / 4

def observe():
    for i in xrange(popsize):
        for j in xrange(chromlen):
            alpha = np.cos(Q[i][j])
            r = np.random.rand()
            P[i][j] = '0' if r < alpha * alpha else '1'

def repair():
    pass

def evaluate():
    pass

def update():
    pass

def storebest():
    pass

def qiga():
    t = 0
    bestval = -1
    initialize()
    observe()
    repair()
    evaluate()
    storebest()

    while t < maxgen:
        observe()
        repair()
        evaluate()
        update()
        storebest()
        t += 1

    print 'fitness: %f\n' % bestval


REPEAT = 3

if __name__ == '__main__':
    print 'qiga'

    start_tm = time.time()

    for rep in xrange(REPEAT):
        qiga()

    stop_tm = time.time()

    print '%g seconds\n' % (stop_tm - start_tm)

