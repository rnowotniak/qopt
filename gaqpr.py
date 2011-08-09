#!/usr/bin/python

import qiga
import random

class GAQPR(qiga.QIGA):
    '''Genetic Algorithm with Quantum Probability Representation (Bin, Junan, et al, 2003)'''

    def __init__(self):
        qiga.QIGA.__init__(self)
        print 'GAQPR constructor'
        self.Pc = 0.7

    def operators(self):
        qiga.QIGA.operators(self)
        if random.random() < self.Pc:
            self.crossover()

    def crossover(self):
        # select two chromosomes with prob. Pc
        # exchange their CETs
        # update
        # change back their CETs

        while True:
            ind1 = random.choice(self.population)
            ind2 = random.choice(self.population)
            if ind1 is not ind2:
                break

        tmp = ind1.cet
        ind1.cet = ind2.cet
        ind2.cet = tmp

        qiga.update(ind1, self.best)
        qiga.update(ind2, self.best)

        tmp = ind1.cet
        ind1.cet = ind2.cet
        ind2.cet = tmp

