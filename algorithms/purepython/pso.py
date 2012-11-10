#!/usr/bin/python

import qopt.framework as framework
import random, copy
import numpy as np

class PSO(framework.EA):
    """Particle Swarm Optimization Algorithm"""

    def __init__(self):
        framework.EA.__init__(self)
        print 'PSO constructor'
        self.c1 = 2
        self.c2 = 2
        self.maxVelocity = None
        self.dim = 2
        self.bounds = []
        self.gbest = None
        self.tmax = 200

    def initialize(self):
        assert len(self.bounds) == self.dim
        if self.maxVelocity == None:
            self.maxVelocity = 0.02 * \
                    np.sqrt(np.square([self.bounds[j][1] - self.bounds[j][0] for j in xrange(self.dim)]).sum())
                    # 0.02 * pierwiastek sumy kwadratow rozmiarow dziedziny w poszczegolnych wymiarach
                    # (zatem czastka bedzie potrzebowala maksymalnie 50 iteracji by przemierzyc rozpietosc dziedziny)
        self.population = []
        for i in xrange(self.popsize):
            p = framework.Individual()
            p.genotype = np.random.rand(1, self.dim)[0]
            for j in xrange(self.dim):
                p.genotype[j] *= (self.bounds[j][1] - self.bounds[j][0])
                p.genotype[j] += self.bounds[j][0]
            # sqrt( dim * bok^2 ) = maxVel  =>   bok = maxVel / sqrt(dim)   (2.* is due to -.5)
            p.velocity = np.array([2. * self.maxVelocity/np.sqrt(self.dim)*(random.random() - .5) for d in xrange(self.dim)])
            p.pbest = None
            self.population.append(p)

    def operators(self):
        for p in self.population:
            if not p.pbest or self.minmaxop(p.fitness, p.pbest.fitness):
                c = copy.deepcopy(p)
                del c.pbest
                p.pbest = c
            if not self.gbest or self.minmaxop(p.fitness, self.gbest.fitness):
                c = copy.deepcopy(p)
                del c.pbest
                self.gbest = c

        for p in self.population:
            # update the particle velocity and position
            p.velocity += self.c1 * random.random() * (p.pbest.genotype - p.genotype) + \
                    self.c2 * random.random() * (self.gbest.genotype - p.genotype)
            velmag = np.sqrt(np.square(p.velocity).sum())
            if velmag > self.maxVelocity:
                p.velocity /= velmag
                p.velocity *= self.maxVelocity
            p.genotype += p.velocity
