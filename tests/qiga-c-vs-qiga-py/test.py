#!/usr/bin/python

import sys

import qigacython

q = qigacython.QIGA()

q.tmax = 500
q.evaluator = qigacython.fknapsack

q.run()

print q.best

print q.Q

#qigacython.start()
