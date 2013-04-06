#!/usr/bin/python -u

import time
import sys
import numpy as np

from multiprocessing import Pool

from random import random, seed

import qopt.problems
# from qopt.algorithms import MyRQIEA2
from qopt.algorithms import MyRQIEA2

from pybrain.optimization import CMAES, GA, PGPE, OriginalNES, NelderMead
from pybrain.optimization import ParticleSwarmOptimizer as PSO

import logging
logging.disable(logging.WARNING)

REPEAT = 25
DIM = 2
MaxFE = 10000 * DIM # according to CEC2013
#MaxFE = 2000

DATA = np.matrix(np.zeros((28, 4 * 4)))  # 2 algs * 4 vals

# pool = Pool(processes = 10)

def PSO_Factory():
    return PSO(boundaries=[[-100,100]]*DIM)

def CMAES_Factory():
    return CMAES()

def GA_Factory():
    return GA()

def PGPE_Factory():
    return PGPE()

def NES_Factory():
    return OriginalNES()

def NelderMead_Factory():
    return NelderMead()


# np.random.seed(1)

def runAlg(algFactory):
    results = []
    t0 = time.time()
    for _ in xrange(REPEAT):
        alg = algFactory()
        alg.numParameters = DIM
        alg.setEvaluator(pyBrainFunWrapper)
        alg.minimize = True
        alg.maxEvaluations = MaxFE
        try:
            best = alg.learn()
        except Exception:
            best = [None, 1000]
        results.append(best[1])
    t1 = time.time()
    results = np.array(results)
    m = np.matrix(np.zeros((1,4)))
    m[0,0] = results.mean()
    m[0,1] = np.median(results)
    m[0,2] = results.std()
    m[0,3] = (t1-t0) / REPEAT
    print '%12g %12g' % (results.mean(), results.std()),
    return m

for fnum in xrange(1,29):
    fobj = qopt.problems.CEC2013(fnum)
    optval = fobj.evaluate(fobj.optimum)

    print '%2d %5g' % (fnum, optval),

    def pyBrainFunWrapper(x):
        return fobj.evaluate(x.tolist())

    DATA[fnum - 1,  0: 4] = runAlg(PSO_Factory)
    DATA[fnum - 1,  4: 8] = runAlg(CMAES_Factory)
    DATA[fnum - 1,  8:12] = runAlg(GA_Factory)
    DATA[fnum - 1, 12:16] = runAlg(NelderMead_Factory)
    print

print '---------------------------------------------------'

np.save('pso-cmaes-ga-nelder-dim%d.npy' % DIM, DATA)
np.savetxt('pso-cmaes-ga-nelder-dim%d.txt' % DIM, DATA)

for fnum in xrange(1,29):
    print '%2d' % fnum,
    print ('%10g ' * (4*4)) % tuple(DATA[fnum-1].tolist()[0])  # algs * vals

