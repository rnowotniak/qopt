#!/usr/bin/python

import sys
from numpy import pi

import random
import qopt.problems

cassini = qopt.problems.CEC2011(15)

best_known = [-779.629801566988, 3.265804135361, 0.528440291493, 0.382390419772,\
                167.937610148996, 424.032204472497, 53.304869390732, 589.767895836123, 2199.961911685212,\
                        0.772877728290, 0.531757418755, 0.010789195916, 0.167388829033, 0.010425709182,\
                                1.358596310427, 1.050001151443, 1.306852313623, 69.813404643644, -1.593310577644,\
                                        -1.959572311812, -1.554796022348, -1.513432303179]

best_known_val = 8.383 # cassini.evaluate(best_known)

bounds = [
        [-1000  ,  0],
        [3      ,  5],
        [0      ,  1],
        [0      ,  1],
        [100    ,  400],
        [100    ,  500],
        [30     ,  300],
        [400    ,  1600],
        [800    ,  2200],
        [0.01   ,  0.9],
        [0.01   ,  0.9],
        [0.01   ,  0.9],
        [0.01   ,  0.9],
        [0.01   ,  0.9],
        [1.05   ,  6],
        [1.05   ,  6],
        [1.15   ,  6.5],
        [1.7    ,  291],
        [-pi    ,  pi],
        [-pi    ,  pi],
        [-pi    ,  pi],
        [-pi    ,  pi]]
ranges = [1.0 * (ub-lw) for (lw,ub) in bounds]

sigma = .01

if len(sys.argv) > 1:
    sigma = float(sys.argv[1])

while True:
    x = []
    for i in xrange(len(best_known)):
        while True:
            p = random.gauss(best_known[i], ranges[i] * sigma)
            if p >= bounds[i][0] and p <= bounds[i][1]:
                break
        x.append(p)
    val = cassini.evaluate(x)
    if val < best_known_val:
        print x
        print val


