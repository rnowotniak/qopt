#!/usr/bin/python

import sys
import types
import qigacython

class QIGA(qigacython.QIGA):
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
q.problem = qigacython.KnapsackProblem()

q.run()
print q.bestval

# qigacython.testtime(q)

sys.exit(0)

q._initialize()
print q.Q

