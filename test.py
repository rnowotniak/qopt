#!/usr/bin/python
#
# General test script of the whole QOpt framework
#

import sys

import qopt.algorithms
import qopt.problems

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

q = QIGA()
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


