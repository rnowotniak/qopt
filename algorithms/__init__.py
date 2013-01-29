
from _algorithms import QIGA, BQIGAo, BQIGAo2, QIGA_StorePriorToRepair

from _myrqiea2 import MyRQIEA2

#from _algorithms import RQIEA
#...



def SGA(evaluator, chromlen, popsize = 50, elitism = False):
    from pyevolve import G1DList,G1DBinaryString
    from pyevolve import GSimpleGA,Selectors,Consts,Scaling
    genome = G1DBinaryString.G1DBinaryString(chromlen)
    genome.evaluator.set(evaluator)
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setElitism(elitism)
    ga.setPopulationSize(popsize)
    #ga.setGenerations(160)
    #ga.setCrossoverRate(.65) #
    ga.setMutationRate(.02) # .02 na poziomie chromosomu i na poziomie populacji (to to samo!)
    #ga.setMutationRate(0)
    ga.setSortType(Consts.sortType["raw"]) # !!!
    ga.selector.set(Selectors.GRouletteWheel) # !!!
    ga.setMinimax(Consts.minimaxType["maximize"])
    #pop = ga.getPopulation()
    #pop.scaleMethod.set(Scaling.SigmaTruncScaling)  # not used due to raw sorting
    return ga


