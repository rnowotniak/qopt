#!/usr/bin/python
#
# Algorithm rQIEA from section 2.3 in Gexiang's Survey
#

import sys
import math
import numpy as np
import random

random.seed(1)
np.random.seed(1)

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

popsize = 50
dim = 30

range_ = (-100,100)

Q = np.zeros([popsize, 2, dim])
P = np.zeros([popsize, dim])


t = 0

# Initialize Q(t)
for i in xrange(popsize):
    Q[i][0] = np.random.random((1, dim)) * 2 - 1
    Q[i][1] = np.sqrt(1-Q[i][0]**2)


minfitness = float('inf')
b = None
termination = False
NoFE = 0
MaxNoFE = 1.5e5

while not termination:
    # Construct P(t) -- very specific to rQIEA algorithm
    for i in xrange(popsize):
        P[i] = range_[0] + np.array([q[random.choice((0,1))] for q in (Q[i]**2).transpose()]) \
                * (range_[1] - range_[0])

    # Evaluate P(t)
    midminfitness = float('inf')
    for i in xrange(popsize):
        fitness = evaluate(P[i])
        if fitness < midminfitness:
            midminfitness = fitness
            angle = np.matrix(Q[i], copy=True)
            midx = P[i]

    # Store the best solution in b
    if midminfitness < minfitness:
        minfitness = midminfitness
        anglemin = angle
        b = np.matrix(midx, copy=True)


    # check termination
    NoFE += popsize
    if NoFE >= MaxNoFE:
        termination = True
        continue

    # Update Q(t) using Q-gates
    for i in xrange(popsize):
        for j in xrange(dim):
            aa = Q[i][0][j]
            bb = Q[i][1][j]
            angleaa = anglemin[0,j]
            anglebb = anglemin[1,j]
            k = np.pi / (100 + np.mod(t, 100))
            theta = k * rotation(aa, bb, angleaa, anglebb)
            G = np.matrix([ \
                    [np.cos(theta), -np.sin(theta)], \
                    [np.sin(theta),  np.cos(theta)]])
            Q[i][:,j] = np.dot(G, np.matrix(Q[i])[:,j]).transpose() # actual rotation

    # Recombination
    Pc = 0.05
    for i in xrange(popsize):
        if random.random() < Pc:
            q1 = random.randint(0, popsize - 1)
            q2 = random.randint(0, popsize - 1)
            h1 = random.choice(xrange(dim + 1))
            h2 = random.choice(xrange(dim + 1))
            if h2 < h1:
                h1, h2 = h2, h1
            temp = np.matrix(Q[q1], copy=True)[:,h1:h2]
            np.matrix(Q[q1], copy=False)[:,h1:h2] = np.matrix(Q[q2])[:,h1:h2] # possibly swap alphas to betas also here
            np.matrix(Q[q2], copy=False)[:,h1:h2] = temp

    t += 1

print b
print minfitness


