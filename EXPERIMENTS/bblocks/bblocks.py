#!/usr/bin/python

import sys

import qopt.algorithms
import qopt.problems

class QIGA(qopt.algorithms.QIGA):
    def generation(self):
        super(QIGA, self).generation()
        #bbs = bblocks(self.P, '***101*1******')
        bbs = 5
        print '%d %f %d' % (self.t, self.bestval, bbs)

qiga = QIGA(chromlen = 20)
qiga.tmax = 160
qiga.problem = qopt.problems.func1d.f3
qiga.run()
print qopt.problems.func1d.f3.getx(qiga.best[:20])
#print qiga.best[:20]


