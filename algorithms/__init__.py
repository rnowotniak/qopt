
from _algorithms import QIGA

#from _algorithms import RQIEA
#...



def SGA(evaluator, chromlen):
    from pyevolve import G1DList,G1DBinaryString
    from pyevolve import GSimpleGA,Selectors,Consts,Scaling
    genome = G1DBinaryString.G1DBinaryString(chromlen)
    genome.evaluator.set(evaluator)
    ga = GSimpleGA.GSimpleGA(genome)
    ga.setPopulationSize(10)
    ga.selector.set(Selectors.GRouletteWheel)
    ga.setGenerations(160)
    ga.setMutationRate(0.005)
    ga.setElitism(False)
    ga.setSortType(Consts.sortType["raw"])
    ga.setMinimax(Consts.minimaxType["maximize"])
    pop = ga.getPopulation()
    pop.scaleMethod.set(Scaling.SigmaTruncScaling)
    return ga


