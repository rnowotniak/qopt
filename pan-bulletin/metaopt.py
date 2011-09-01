#!/usr/bin/python

from pyevolve import GSimpleGA
from pyevolve import G1DList
from pyevolve import Selectors
from pyevolve import Consts
from pyevolve import DBAdapters
from pyevolve import Initializators, Mutators

import subprocess,sys,os
import math

def getPhenotype(genome):
    return (genome[0] * 0.1, genome[1] * 0.1, genome[2] * 50000, genome[3] * 50000)

def eval_func(genome):
    s = ' '.join([str(x) for x in genome])
    while True:
        log.write('> %s\n' % s)
        proc.stdin.write('%s\n' % s)
        line = proc.stdout.readline()
        log.write('< ' + line)
        result = float(line)
        if result > 0 and result < 2000:
            break
    return result

#os.environ['LD_LIBRARY_PATH'] = '/opt/MatlabR2008b/bin/glnx86'
proc = subprocess.Popen('./evaluator-qiga-prerand-curand-tuning', shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
log = open('/tmp/log.%d'%os.getpid(), 'w', 0)

# print 'val: ', eval_func([0.058998872121786061, 1, 0.0092901516939887241, 0.14086122789968736])
# subprocess.Popen('kill -9 ' + str(proc.pid), shell=True)
# sys.exit(0)

#subprocess.Popen('killall forConsole', shell=True)
#sys.exit(0)

def run_main():
    # Genome instance
    genome = G1DList.G1DList(5)
    genome.setParams(rangemin=0.0 * math.pi / 180, rangemax= 15.0 * math.pi / 180, gauss_sigma = 5 * math.pi / 180)

    # Change the initializator to Real values
    genome.initializator.set(Initializators.G1DListInitializatorReal)

    # Change the mutator to Gaussian Mutator
    genome.mutator.set(Mutators.G1DListMutatorRealGaussian)


    # The evaluator function (objective function)
    genome.evaluator.set(eval_func)

    # Genetic Algorithm Instance
    ga = GSimpleGA.GSimpleGA(genome)
    sqlite_adapter = DBAdapters.DBSQLite(identify=sys.argv[1])                                                                                                                                                        
    ga.setDBAdapter(sqlite_adapter)   

    ga.selector.set(Selectors.GRankSelector)
    ga.setSortType(Consts.sortType["raw"])
    ga.setGenerations(50)
    ga.setPopulationSize(10)
    ga.setMutationRate(1./5/3)
    ga.setMinimax(Consts.minimaxType['maximize'])
    print ga

    # Do the evolution
    ga.evolve(freq_stats=1)

    # Best individual
    print ga.bestIndividual()

if __name__ == "__main__":
    run_main()
    subprocess.Popen('killall -9 evaluator-qiga-prerand-curand-tuning', shell=True)

#
# val:  0.0 0.00891966286553 3798.19270026 23713.6596329
# 333.7699
#
#
#

