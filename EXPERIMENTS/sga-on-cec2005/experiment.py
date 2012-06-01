#!/usr/bin/python

import sys
import math

import qopt.benchmarks.CEC2005.cec2005 as cec2005

###########################
FNUM = 5 # XXX
eps = 10**-8
dim = 10 # XXX -> 30
Max_FES = 10000 * dim
popsize = 100
###########################



bounds = cec2005.FUNCTIONS['f1']['bounds']
domain = bounds[1] - bounds[0]
width = int(math.ceil(math.log(domain/10**-8) / math.log(2)))
print 'Kazdy parametr kodowany na %d binarnych genach' % width

from pyevolve import G1DList,G1DBinaryString
from pyevolve import GSimpleGA,Selectors,Consts,Scaling
import numpy as np
import qopt.benchmarks.CEC2005.cec2005 as cec2005

bfunc = cec2005.FUNCTIONS['f' + str(FNUM)]

def getGeno(string):
    return [bfunc['bounds'][0] + 1.0 * domain * int(string[width*i:width*(i+1)], 2)/(2.**width-1) for i in xrange(dim)]

def eval_func(chrom):
    chrom = ''.join([str(c) for c in chrom[:]])
    return bfunc['fun'](getGeno(chrom))

def step_func(ga):
    print '%d %g' % (ga.getCurrentGeneration(), ga.bestIndividual().getRawScore())

genome = G1DBinaryString.G1DBinaryString(width * dim)
genome.evaluator.set(eval_func)
ga = GSimpleGA.GSimpleGA(genome)
ga.setPopulationSize(popsize)
ga.selector.set(Selectors.GRouletteWheel)
ga.setGenerations(Max_FES / popsize)
ga.setMutationRate(0.05)
ga.setElitism(True)
ga.setSortType(Consts.sortType["raw"])
ga.setMinimax(Consts.minimaxType["minimize"])

ga.stepCallback.set(step_func)

pop = ga.getPopulation()
pop.scaleMethod.set(Scaling.SigmaTruncScaling)
# print ga

ga.evolve(freq_stats=0)
#print getGeno(''.join([str(c) for c in ga.bestIndividual()[:]]))



