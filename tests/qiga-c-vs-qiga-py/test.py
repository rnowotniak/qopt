#!/usr/bin/python

import sys

import qigacython

q = qigacython.QIGA()
q.tmax = 500
q.problem = qigacython.KnapsackProblem()

#q.initialization += 

#q.initialize()

#print q.Q

q.run()

print q.bestval

#print q.Q





#qigacython.runcpp()

#qigacython.start()

