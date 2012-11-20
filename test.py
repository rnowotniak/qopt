#!/usr/bin/python
# coding=utf-8
#
# General test script of the whole QOpt framework
#

import sys, os

import qopt.algorithms
import qopt.problems

import pylab


############
# Problems #
############


# knapsack
knapsack = qopt.problems.knapsack
print knapsack.evaluate('1' * 250)

#sys.exit(0)

# func 1d
f1d = qopt.problems.func1d.f1
print f1d.evaluate(60.488)

# sat
import qopt.problems._sat
s1 = qopt.problems._sat.SatProblem('problems/sat/flat30-100.cnf')
print s1.evaluate('100100001100100100001100010100010001010010100010100010010010010010001001001100001001001001')

# cec2005
f1 = qopt.problems.cec2005(1)
print f1.evaluate((0,0))

cassini = qopt.problems.cec2011(15) # XXX
print cassini.evaluate([-779.629801566988, 3.265804135361, 0.528440291493, 0.382390419772,\
        167.937610148996, 424.032204472497, 53.304869390732, 589.767895836123, 2199.961911685212,\
        0.772877728290, 0.531757418755, 0.010789195916, 0.167388829033, 0.010425709182,\
        1.358596310427, 1.050001151443, 1.306852313623, 69.813404643644, -1.593310577644,\
        -1.959572311812, -1.554796022348, -1.513432303179])
# should be 8.383...

# cec2011

##############################
# Combinatorial optimization #
##############################

class QIGA(qopt.algorithms.QIGA):
    def initialize(self):
        super(QIGA, self).initialize()
        print 'my initialization'
        print self.Q

    def generation(self):
        super(QIGA, self).generation()
        if self.t == 5:
            print 'generation %d, bestval: %g' % (self.t, self.bestval)

q = QIGA(chromlen = 250)
q.tmax = 500
q.problem = qopt.problems.knapsack
q.run()
print q.best
print q.bestval
#print q.P[0]
#print q.Q[3,5]

#r = qopt.algorithms.rQIEA()
sys.exit(0)

q._initialize()
print q.Q


########################################
# Real-Coded -- numerical optimization #
########################################

# cec2005
r = qopt.algorithms.RQIEA
r.problem = qopt.problems.cec2005.f2
r.dim = 30
r.bounds = None # ??? set automatically in .problem ?
r.run()

# cec2011
r.problem = qopt.problems.cec2011.f15
r.run()


