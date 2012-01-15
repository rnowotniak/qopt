#!/usr/bin/python
#
# Algorithm rQIEA from section 2.3 in Gexiang's Survey
#

import sys
import math
import numpy as np
import random

# random.seed(1)
# np.random.seed(1)

def evaluate(x):
    # sphere function
    return sum(np.array(x)**2)

def rotation(aa, bb, angleaa, anglebb):
    xi_b = math.atan(anglebb / angleaa);
    xi_ij = math.atan(bb / aa);
    
    if xi_b > 0 and xi_ij > 0:
        if xi_b >= xi_ij:
            return 1
        else:
            return -1
    elif xi_b > 0 and xi_ij <= 0:
        return np.sign(angleaa * aa)
    elif xi_b <= 0 and xi_ij > 0:
        return -np.sign(angleaa * aa)
    elif xi_b <= 0 and xi_ij <= 0:
        if xi_b >= xi_ij:
            return 1
        else:
            return -1
    elif xi_b == 0 or xi_ij == 0 or abs(xi_b - np.pi/2) < 0.001 or abs(xi_b - np.pi/2) < 0.001:
        return np.sign(random.random() - .5)
    else:
        print 'error in rotation'


class rQIEA():

    def __init__(self):
        self.popsize = 20
        self.dim = 10
        # XXX range_ -> bounds
        self.range_ = (-100,100)
        self.iter = 0
        self.minfitness = float('inf')
        self.b = None
        self.termination = False
        self.NoFE = 0
        self.MaxNoFE = 1e5

    def initialize(self):
        self.Q = np.zeros([self.popsize, 2, self.dim])
        self.P = np.zeros([self.popsize, self.dim])
        # Initialize Q(self.iter)
        for i in xrange(self.popsize):
            self.Q[i][0] = np.random.random((1, self.dim)) * 2 - 1
            self.Q[i][1] = np.sqrt(1-self.Q[i][0]**2)

    # XXX -> step
    def run(self):
        while not self.termination:
            # Construct P(self.iter) -- very specific to rQIEA algorithm
            for i in xrange(self.popsize):
                self.P[i] = self.range_[0] + np.array([q[random.choice((0,1))] for q in (self.Q[i]**2).transpose()]) \
                        * (self.range_[1] - self.range_[0])

            # Evaluate P(self.iter)
            midminfitness = float('inf')
            for i in xrange(self.popsize):
                fitness = self.evaluate(self.P[i])
                if fitness < midminfitness:
                    midminfitness = fitness
                    angle = np.matrix(self.Q[i], copy=True)
                    midx = self.P[i]

            # Store the best solution in self.b
            if midminfitness < self.minfitness:
                self.minfitness = midminfitness
                anglemin = angle
                self.b = np.matrix(midx, copy=True)


            # check termination
            self.NoFE += self.popsize
            if self.NoFE >= self.MaxNoFE:
                self.termination = True
                continue

            # Update Q(self.iter) using Q-gates
            for i in xrange(self.popsize):
                for j in xrange(self.dim):
                    aa = self.Q[i][0][j]
                    bb = self.Q[i][1][j]
                    angleaa = anglemin[0,j]
                    anglebb = anglemin[1,j]
                    k = np.pi / (100 + np.mod(self.iter, 100))
                    theta = k * rotation(aa, bb, angleaa, anglebb)
                    G = np.matrix([ \
                            [np.cos(theta), -np.sin(theta)], \
                            [np.sin(theta),  np.cos(theta)]])
                    self.Q[i][:,j] = np.dot(G, np.matrix(self.Q[i])[:,j]).transpose() # actual rotation

            # Recombination
            Pc = 0.05
            for i in xrange(self.popsize):
                if random.random() < Pc:
                    q1 = random.randint(0, self.popsize - 1)
                    q2 = random.randint(0, self.popsize - 1)
                    h1 = random.choice(xrange(self.dim + 1))
                    h2 = random.choice(xrange(self.dim + 1))
                    if h2 < h1:
                        h1, h2 = h2, h1
                    temp = np.matrix(self.Q[q1], copy=True)[:,h1:h2]
                    np.matrix(self.Q[q1], copy=False)[:,h1:h2] = np.matrix(self.Q[q2])[:,h1:h2] # possibly swap alphas to betas also here
                    np.matrix(self.Q[q2], copy=False)[:,h1:h2] = temp

            self.iter += 1

        return [self.b, self.minfitness]


if __name__ == '__main__':
    rqiea = rQIEA()
    rqiea.initialize()
    print rqiea.run()

