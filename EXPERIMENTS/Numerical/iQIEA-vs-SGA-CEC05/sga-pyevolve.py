#!/usr/bin/python

import sys
from pyevolve import G1DList,G1DBinaryString
from pyevolve import GSimpleGA,Selectors,Consts,Scaling
import numpy as np
import qopt.benchmarks.CEC2005.cec2005 as cec2005

FNUM = sys.argv[1]
dim = 10
bfunc = cec2005.FUNCTIONS['f' + str(FNUM)]

def eval_func(chrom):
    chrom = ''.join([str(c) for c in chrom[:]])
    wid = bfunc['bounds'][1] - bfunc['bounds'][0]
    x = [bfunc['bounds'][0] + wid * int(chrom[30*i:30*(i+1)], 2)/(2.**30-1) for i in xrange(dim)]
    return bfunc['fun'](x)

genome = G1DBinaryString.G1DBinaryString(30 * dim)
genome.evaluator.set(eval_func)
ga = GSimpleGA.GSimpleGA(genome)
ga.setPopulationSize(100)
ga.selector.set(Selectors.GRouletteWheel)
ga.setGenerations(100)
ga.setMutationRate(0.05)
ga.setElitism(True)
ga.setSortType(Consts.sortType["raw"])
ga.setMinimax(Consts.minimaxType["minimize"])
pop = ga.getPopulation()
pop.scaleMethod.set(Scaling.SigmaTruncScaling)

# print ga

ga.evolve(freq_stats=1)

