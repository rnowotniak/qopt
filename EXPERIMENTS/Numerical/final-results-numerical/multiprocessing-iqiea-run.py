#!/usr/bin/python -u

import sys
import numpy as np

from random import random
import time

from multiprocessing import Pool

import qopt.problems
from qopt.algorithms import iQIEA

FUNCS = 28
REPEAT = 10
DIM = 2
MaxFE = 10000 * DIM # according to CEC2013
#MaxFE = 2000
qiea_popsize = 10

alg = iQIEA()

# iqiea.fitness_function = cec2013wrapper
alg.G = DIM
alg.bounds = [(-100,100)]*DIM
alg.popsize = qiea_popsize
alg.K = 15
alg.kstep = 20
alg.tmax = MaxFE / alg.K
alg.XI = .3
alg.DELTA = .3




print ' f   opt  |      mean       stddev |'
print '------------------------------------'

table = np.matrix(np.zeros((28,4)))

def runAlg(fnum):
    fobj = qopt.problems.CEC2013(fnum)
    optval = fobj.evaluate(fobj.optimum)

    def cec2013wrapper(x):
        return fobj.evaluate(x)

    # myrqiea2
    alg.fitness_function = cec2013wrapper
    results = []
    t0 = time.time()
    for _ in xrange(REPEAT):
        alg.run()
        results.append(alg.bestval)
    t1 = time.time()
    results = np.array(results)
    row = np.matrix('0. 0. 0. 0.')
    row[0,0] = results.mean()
    row[0,1] = np.median(results)
    row[0,2] = results.std()
    row[0,3] = (t1-t0) / REPEAT
    print '%2d %5g' % (fnum, results.mean()),
    return row
    # print '%12g %12g %12g %12g' % tuple(row.tolist()[0])
    # myrqiea2 done

pool = Pool(processes = 28)
table = np.matrix( np.vstack(pool.map(runAlg, xrange(1,FUNCS+1))) )

for fnum in xrange(1,FUNCS+1):
    print '%2d     ' % fnum,
    print ('%12g ' * (1*4)) % tuple(table[fnum-1].tolist()[0])  # algs * vals

print '---------------------------------------------------'

np.save('/tmp/multiprocessing-iqiea-dim%d.npy' % DIM, table)
np.savetxt('/tmp/multiprocessing-iqiea-dim%d.txt' % DIM, table)

