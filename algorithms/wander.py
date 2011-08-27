#!/usr/bin/python

# testowy algorytm, losowego bladzenia

import ea,sys,random

class Wander(ea.EA):
    def __init__(self):
        ea.EA.__init__(self)

    def __rand(self):
        self.population = []
        for ind in xrange(self.popsize):
            ind = ea.Individual()
            if self.repr == 'binary':
                g = (''.join([random.choice(('0','1')) for locus in xrange(10)]),\
                        ''.join([random.choice(('0','1')) for locus in xrange(10)]))
            else: # 'real'
                g = (random.random(), random.random())
            ind.genotype = g
            self.population.append(ind)


    def initialize(self):
        self.__rand()

    def operators(self):
        self.__rand()

    def evaluation(self):
        for ind in self.population:
            ind.fitness = self.evaluator.eval(*ind.genotype) # <- knapsack etc  // self.evaluator.eval(..) ...?


if __name__ == '__main__':
    wander = Wander()
    wander.parseArgs(sys.argv[1:])
    wander.run()


