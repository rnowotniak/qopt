#!/usr/bin/python

import sys
import numpy as np

from qopt.algorithms.pso import PSO
import qopt.benchmarks.CEC2005.cec2005 as cec2005

pso = PSO()

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def step(ea):
    data = np.array([ind.genotype for ind in ea.population])
    plt.cla()
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.grid(True)
    plt.title('Schifted Rosenbrock\'s Function')
    x = np.arange(-100,100, 1)
    y = np.arange(-100,100, 1)
    x, y = np.meshgrid(x,y)
    z = 0. * x
    for i in xrange(z.shape[0]):
        for j in xrange(z.shape[1]):
            z[i,j] = cec2005.FUNCTIONS[FNUM]['fun']((x[i,j],y[i,j]))
    plt.contourf(x, y, z, 50, cmap = cm.hot)
    plt.plot(data[:,0], data[:,1], 'bo')
    #plt.show()
    plt.savefig('/tmp/f6-pso%05d.png' % ea.evaluation_counter, bbox_inches = 'tight')
    if ea.best:
        print ea.evaluation_counter, ea.best.fitness

# XXX ###############
FNUM = 'f6'
pso.evaluator = lambda ind: cec2005.FUNCTIONS[FNUM]['fun'](ind.genotype)
pso.dim = 2
pso.popsize = 20
pso.tmax = 50
pso.bounds = [ cec2005.FUNCTIONS[FNUM]['bounds'] ] * pso.dim
pso.stepCallback = step
#####################

print pso.bounds

pso.run()
print pso.best

