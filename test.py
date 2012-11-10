#!/usr/bin/python

import sys

import qopt.problems.knapsack as knapsack
# import qopt.problems.tsp
# import qopt.problems.sat
# import qopt.problems.func1d
# import qopt.problems.cec2005
# import qopt.problems.cec2011

import qopt.algorithms

class QIGA(qopt.algorithms.BLAA):
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
q.problem = knapsack.KnapsackProblem()

q.run()
print q.bestval

# qigacython.testtime(q)

sys.exit(0)

q._initialize()
print q.Q

