#!/usr/bin/python -u

import sys
import numpy as np

from random import random

import qopt.problems
# from qopt.algorithms import MyRQIEA2
from qopt.algorithms import MyRQIEA2

from pybrain.optimization import CMAES
from pybrain.optimization import ParticleSwarmOptimizer as PSO

import logging
logging.disable(logging.WARNING)

REPEAT = 10
DIM = 2
#MaxFE = 10000 * DIM # according to CEC2013
MaxFE = 1000
qiea_popsize = 10

myrqiea2 = MyRQIEA2(chromlen = DIM, popsize = qiea_popsize)
myrqiea2.tmax = MaxFE / qiea_popsize

print '          |        MyRQIEA-2       |                 PSO            |           CMAES'
print ' f   opt  |      mean       stddev |         mean          stddev   |     mean         stddev'
print '----------------------------------------------------------------------------------------------'

for fnum in xrange(1,29):
    fobj = qopt.problems.CEC2013(fnum)
    optval = fobj.evaluate(fobj.optimum)

    # myrqiea2
    myrqiea2.problem = fobj
    results = []
    for r in xrange(REPEAT):
        myrqiea2.run()
        results.append(myrqiea2.bestval)
    results = np.array(results)
    print '%2d %5g' % (fnum, optval),
    print '%12g %12g' % (results.mean(), results.std()),
    print '    ',
    # myrqiea2 done

    def pyBrainFunWrapper(x):
        return fobj.evaluate(x.tolist())

    # PSO
    results = []
    for r in xrange(REPEAT):
        alg = PSO(boundaries=[[-100,100]]*DIM)
        alg.numParameters = DIM
        alg.setEvaluator(pyBrainFunWrapper)
        alg.minimize = True
        alg.maxEvaluations = MaxFE
        best = alg.learn()
        results.append(best[1])
    results = np.array(results)
    print '%12g %12g' % (results.mean(), results.std()),
    # PSO done

    # CMAES
    results = []
    for r in xrange(REPEAT):
        alg = CMAES()
        alg.numParameters = DIM
        alg.setEvaluator(pyBrainFunWrapper)
        alg.minimize = True
        alg.maxEvaluations = MaxFE
        best = alg.learn()
        results.append(best[1])
    results = np.array(results)
    print '%12g %12g' % (results.mean(), results.std())
    # CMAES done



print '---------------------------------------------------'

