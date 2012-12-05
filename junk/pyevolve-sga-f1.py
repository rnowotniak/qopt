#!/usr/bin/python

import sys
import qopt.problems
from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
from pyevolve import Consts

f = qopt.problems.func1d.f1

#print f.evaluate('0' + '1' * 1)

def eval_func(chromo):
    s = ''.join([str(g) for g in chromo])
    return f.evaluate(s) + 1

genome = G1DBinaryString.G1DBinaryString(15)
genome.evaluator.set(eval_func)

sga = GSimpleGA.GSimpleGA(genome)
sga.setPopulationSize(100)
sga.setMutationRate(0)
sga.setGenerations(80)
sga.setElitism(False)
#sga.setSortType(Consts.sortType['raw'])
sga.selector.set(Selectors.GRouletteWheel)


sga.evolve(freq_stats = 1)


print genome
print sga
print sga.getPopulation()
print '----'

print sga.bestIndividual()

