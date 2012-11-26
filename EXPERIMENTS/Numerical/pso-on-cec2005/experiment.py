#!/usr/bin/python

import sys
import numpy as np

from qopt.algorithms.pso import PSO
import qopt.benchmarks.CEC2005.cec2005 as cec2005

pso = PSO()

def step(ea):
    if ea.best:
        print ea.evaluation_counter, ea.best.fitness
        sys.stdout.flush()

# XXX ###############
FNUM = 'f%s' % sys.argv[1]
pso.evaluator = lambda ind: cec2005.FUNCTIONS[FNUM]['fun'](ind.genotype)
pso.dim = 10
pso.popsize = 100
pso.tmax = 1000
pso.bounds = [ cec2005.FUNCTIONS[FNUM]['bounds'] ] * pso.dim
pso.stepCallback = step
#####################

print pso.bounds

pso.run()
print pso.best

