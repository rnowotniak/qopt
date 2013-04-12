#!/usr/bin/python -u

import sys
import numpy as np

from random import random
import time

from multiprocessing import Pool

import qopt.problems
from qopt.algorithms import QIEA1

FUNCS = 28
REPEAT = 25
DIM = 2
MaxFE = 10000 * DIM # according to CEC2013
#MaxFE = 2000
qiea_popsize = 10

qiea1 = QIEA1(chromlen = DIM, popsize = qiea_popsize)
qiea1.tmax = MaxFE / qiea_popsize

print ' f   opt  |      mean       median     stddev'
print '------------------------------------'

table = np.matrix(np.zeros((28,4)))

def runAlg(fnum):
    fobj = qopt.problems.CEC2013(fnum)
    optval = fobj.evaluate(fobj.optimum)

    # qiea1
    qiea1.problem = fobj
    results = []
    t0 = time.time()
    for _ in xrange(REPEAT):
        qiea1.run()
        results.append(qiea1.bestval)
    t1 = time.time()
    results = np.array(results)
    row = np.matrix('0. 0. 0. 0.')
    row[0,0] = results.mean()
    row[0,1] = np.median(results)
    row[0,2] = results.std()
    row[0,3] = (t1-t0) / REPEAT
    return row
    # print '%2d %5g' % (fnum, optval),
    # print '%12g %12g %12g %12g' % tuple(row.tolist()[0])
    # qiea1 done

pool = Pool(processes = 8)
table = np.matrix( np.vstack(pool.map(runAlg, xrange(1,FUNCS+1))) )

for fnum in xrange(1,FUNCS+1):
    print '%2d     ' % fnum,
    print ('%12g ' * (1*4)) % tuple(table[fnum-1].tolist()[0])  # algs * vals

print '---------------------------------------------------'

np.save('/tmp/multiprocessing-qiea1-dim%d.npy' % DIM, table)
np.savetxt('/tmp/multiprocessing-qiea1-dim%d.txt' % DIM, table)

