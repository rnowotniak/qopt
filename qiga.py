#!/usr/bin/python

import framework
import math
import numpy
import random

class QIGA(framework.GA):
    '''Quantum-Inspired Genetic Algorithm (Han, 2000)'''

    def initialize(self):
        if not self.population:
            for i in xrange(self.popsize):
                ind = framework.Individual()
                s2 = math.sqrt(2) / 2
                ind.genotype = numpy.mat(numpy.ones((2,self.chromlen))) * s2
                self.population.append(ind)

    def evaluation(self):
        print 'QIGA evaluating'
        # make P(t) by observing Q(t)
        # evaluate P(t)
        for ind in self.population:
            ind.cet = observe(ind.genotype)
            print ind.cet
            res = self.evaluator(ind.cet)
            if type(res) == tuple:
                ind.fitness, ind.phenotype = res
            else:
                ind.fitness, ind.phenotype = res, None
        
    def operators(self):
        for ind in self.population:
            ind = update(ind, self.best)

def update(ind, best):
    # TODO skonczyc
    #print best.cet, best.fitness
    #print ind.cet, ind.fitness
    for gi in xrange(len(ind.cet)):
        #sgn = numpy.sign(ind.genotype[0,gi]*ind.genotype[1,gi])
        theta = getRotAngle(ind,best,gi)
        U = rot_gate(theta)
        ind.genotype[:,gi] = U * ind.genotype[:,gi] 
        continue
        if best.cet[gi] == '1':
            if ind.cet[gi] == '1':
                if sgn > 0:
                    ind.genotype[:,gi] = rot_add4deg*ind.genotype[:,gi]
                else:
                    ind.genotype[:,gi] = rot_sub4deg*ind.genotype[:,gi]
            else:
                if sgn > 0:
                    ind.genotype[:,gi] = rot_add8deg*ind.genotype[:,gi]
                else:
                    ind.genotype[:,gi] = rot_sub8deg*ind.genotype[:,gi]
        else:
            if ind.cet[gi] == '1':
                if sgn < 0:
                    ind.genotype[:,gi] = rot_add8deg*ind.genotype[:,gi]
                else:
                    ind.genotype[:,gi] = rot_sub8deg*ind.genotype[:,gi]
            else:
                if sgn < 0:
                    ind.genotype[:,gi] = rot_add4deg*ind.genotype[:,gi]
                else:
                    ind.genotype[:,gi] = rot_sub4deg*ind.genotype[:,gi]
    return ind

nan=float('nan')
pi=numpy.pi
lookuptable = numpy.matrix([
    [        0,  0,  0,   0,   0 ],
    [        0,  0,  0,   0,   0 ],
    [        0,  0,  0,   0,   0 ],
    [  0.05*pi, -1, +1, nan,   0 ],
    [  0.01*pi, -1, +1, nan,   0 ],
    [ 0.025*pi, +1, -1,   0, nan ],
    [ 0.005*pi, +1, -1,   0, nan ],
    [ 0.025*pi, +1, -1,   0, nan ]])

def getRotAngle(ind,best,gi):
    xi = int(ind.cet[gi])
    bi = int(best.cet[gi])
    ri = int('%d%d%d' % (xi,bi,ind.fitness>=best.fitness), 2)
    row = lookuptable[ri,:]

    dtheta = row[0,0]
    s = numpy.sign(ind.genotype[0,gi]*ind.genotype[1,gi])
    if s > 0:
        s = row[0,1]
    elif s < 0:
        s = row[0,2]
    elif ind.genotype[0,gi] == 0:
        s = row[0,3]
    else:
        s = row[0,4]
    if s == nan:
        s = 1.0 - 2*random.randint(0,1)
    return dtheta * s

def observe(qgen):
    #print qgen
    return ''.join([str(int(random.random() > qgen[0,i]**2)) for i in xrange(qgen.shape[1])])

# create rotation gate (angle degrees, counterclockwise) in qubit state space
def rot_gate(angle):
    # TODO jakies cacheowanie mozna dodac
    return numpy.matrix(
            [[math.cos(angle), -math.sin(angle)],
                [math.sin(angle), math.cos(angle)]])

if __name__ == '__main__':
    qiga = QIGA()

