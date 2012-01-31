#!/usr/bin/python
#
# Algorithm rQIEA from section 2.3 in Gexiang's Survey
#

import sys
import math
import numpy as np
import random

import copy
import qopt.framework

# XXX
random.seed(1)
np.random.seed(1)

def sphere(x):
    # sphere function
    return sum(np.array(x)**2)

def rotation(alpha, beta, alphabest, betabest):
    xi_b = math.atan(betabest / alphabest);
    xi_ij = math.atan(beta / alpha);
    
    if xi_b > 0 and xi_ij > 0:
        if xi_b >= xi_ij:
            return 1
        else:
            return -1
    elif xi_b > 0 and xi_ij <= 0:
        return np.sign(alphabest * alpha)
    elif xi_b <= 0 and xi_ij > 0:
        return -np.sign(alphabest * alpha)
    elif xi_b <= 0 and xi_ij <= 0:
        if xi_b >= xi_ij:
            return 1
        else:
            return -1
    elif xi_b == 0 or xi_ij == 0 or abs(xi_b - np.pi/2) < 0.001 or abs(xi_b - np.pi/2) < 0.001:
        return random.choice([-1, 1])
    else:
        print 'error in rotation'


class rQIEA(qopt.framework.EA):

    def __init__(self):
        self.popsize = 20
        self.dim = 10
        self.bounds = (-100,100) # it should be an array [(-100, 100), (-100, 100), ...)] XXX
        self.minfitness = float('inf')
        self.b = None
        self.termination = False
        self.NoFE = 0
        self.MaxNoFE = 1e4
        self.Pc = 0.05

    def initialize(self):
        self.Q = np.zeros([self.popsize, 2, self.dim])
        self.P = np.zeros([self.popsize, self.dim])
        # Initialize Q(self.t)
        for i in xrange(self.popsize):
            self.Q[i][0] = np.random.random((1, self.dim)) * 2 - 1
            self.Q[i][1] = np.sqrt(1-self.Q[i][0]**2) # XXX should be positive or negative

    def evaluation(self):  # XXX jak to sie ma do frameworka?
        fvalues = []
        for ind in self.P:
            fvalues.append(self.fitness_function(ind))
        return fvalues

    def updateQ(self):
        # Update Q(self.t) using Q-gates
        for i in xrange(self.popsize):
            for j in xrange(self.dim):
                alpha = self.Q[i][0][j]
                beta = self.Q[i][1][j]
                alphabest = self.bestq[0,j]
                betabest = self.bestq[1,j]
                k = np.pi / (100 + np.mod(self.t, 100))
                theta = k * rotation(alpha, beta, alphabest, betabest)
                G = np.matrix([ \
                        [np.cos(theta), -np.sin(theta)], \
                        [np.sin(theta),  np.cos(theta)]])
                self.Q[i][:,j] = np.dot(G, np.matrix(self.Q[i])[:,j]).transpose() # actual rotation

    def recombination(self):
        # Recombination
        for i in xrange(self.popsize):
            if random.random() < self.Pc:
                q1 = random.randint(0, self.popsize - 1)
                q2 = random.randint(0, self.popsize - 1)
                h1 = random.choice(xrange(self.dim + 1))
                h2 = random.choice(xrange(self.dim + 1))
                if h2 < h1:
                    h1, h2 = h2, h1
                temp = np.matrix(self.Q[q1], copy=True)[:,h1:h2]
                np.matrix(self.Q[q1], copy=False)[:,h1:h2] = np.matrix(self.Q[q2])[:,h1:h2] # possibly swap alphas to betas also here
                np.matrix(self.Q[q2], copy=False)[:,h1:h2] = temp

    def generation(self):
        # Observe, Construct P(self.t) -- very specific to rQIEA algorithm
        for i in xrange(self.popsize):
            for j in xrange(self.dim):
                r = random.random()
                if r <= 0.5:
                    self.P[i,j] = self.Q[i][0][j]**2
                else:
                    self.P[i,j] = self.Q[i][1][j]**2
                # mapping into range of the optimization variable
                self.P[i,j] *= self.bounds[1] - self.bounds[0]
                self.P[i,j] += self.bounds[0]

        # Evaluate P(t)
        fvalues = self.evaluation()

        # Select the best solution and store it into b(t)
        self.best = min(fvalues) # minmax XXX
        self.bestq = copy.deepcopy(self.Q[fvalues.index(self.best)])

        #  # midminfitness = float('inf')
        #  # for i in xrange(self.popsize):
        #  #     fitness = self.fitness_function(self.P[i])
        #  #     if fitness < midminfitness:
        #  #         midminfitness = fitness
        #  #         angle = np.matrix(self.Q[i], copy=True)
        #  #         midx = self.P[i]

        #  # Store the best solution in self.b
        #  if midminfitness < self.minfitness:
        #      self.minfitness = midminfitness
        #      anglemin = angle
        #      self.b = np.matrix(midx, copy=True)

        #  self.best = None # XXX

        #  # check termination
        #  self.NoFE += self.popsize
        #  if self.NoFE >= self.MaxNoFE:
        #      self.termination = True
        #      continue

        self.updateQ()

        self.recombination()
        print self.t


if __name__ == '__main__':
    rqiea = rQIEA()
    # set parameters
    rqiea.popsize = 6
    rqiea.dim = 3
    rqiea.tmax = 10000 / rqiea.popsize
    rqiea.fitness_function = sphere
    print rqiea.run()
    print rqiea.best

