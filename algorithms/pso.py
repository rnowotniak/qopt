#!/usr/bin/python

import framework
import random, copy, numpy

class PSO(framework.EA):
    """Particle Swarm Optimization Algorithm"""

    def __init__(self):
        framework.EA.__init__(self)
        print 'PSO constructor'
        self.c1 = 2
        self.c2 = 2
        self.xmin, self.xmax = -1., 1.
        self.ymin, self.ymax = -1., 1.
        self.zmin, self.zmax = 0., 0.
        self.maxVelocity = 0.05
        self.dimensions = 2
        self.gbest = None
        self.tmax = 200

    def initialize(self):
        self.population = []
        for i in xrange(self.popsize):
            p = framework.Individual()
            p.genotype = numpy.array([ \
                    self.xmin + (self.xmax - self.xmin) * random.random(), \
                    self.ymin + (self.ymax - self.ymin) * random.random(), \
                    self.zmin + (self.zmax - self.zmin) * random.random()])
            p.genotype = p.genotype[:self.dimensions]
            p.velocity = numpy.array([self.maxVelocity*(random.random() - .5) for d in xrange(self.dimensions)])
            p.pbest = None
            self.population.append(p)

    def operators(self):
        for p in self.population:
            if not p.pbest or p.fitness > p.pbest.fitness:
                p.pbest = copy.deepcopy(p)
            if not self.gbest or p.fitness > self.gbest.fitness:
                self.gbest = copy.deepcopy(p)

        for p in self.population:
            # update the particle velocity and position
            p.velocity += self.c1 * random.random() * (p.pbest.genotype - p.genotype) + \
                    self.c2 * random.random() * (self.gbest.genotype - p.genotype)
            velmag = numpy.sqrt(numpy.square(p.velocity).sum())
            if velmag > self.maxVelocity:
                p.velocity /= velmag
                p.velocity *= self.maxVelocity
            p.genotype += p.velocity
