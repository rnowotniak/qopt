#!/usr/bin/python

import sys

import qopt.algorithms
import qopt.problems


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
print q.bestval
print q.P[0]
print q.Q[3,5]

#r = qopt.algorithms.rQIEA()


sys.exit(0)

q._initialize()
print q.Q

