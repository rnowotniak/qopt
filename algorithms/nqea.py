#!/usr/bin/python

import qiga
import math

class NQEA(qiga.QIGA):

    def __init__(self):
        print 'NQEA constructor'
        self.globalbest = None
        self.l0 = 5
        self.theta1 = 0.01 * math.pi
        self.theta2 = 0.01 * math.pi

    def evaluation(self):
        for ind in self.population:
            genbest = None
            for m in xrange(self.l0):
                c = qiga.sample(ind.genotype)
                f = eval(c) # eval <- knapsack etc
                if f > genbest:
                    genbest = f
            ind.genbest = genbest
            if genbest > ind.pbest:
                ind.pbest = genbest
            if ind.pbest > self.globalbest:
                self.globalbest = ind.pbest

    def operators(self):
        for ind in self.population:
            # apply the restricted update operator to ind
            for i in xrange(len(ind.genotype[0])):
                qiga.rotate(....)


