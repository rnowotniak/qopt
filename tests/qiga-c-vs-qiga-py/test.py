#!/usr/bin/python

import sys

import qigacython

q = qigacython.QIGA()
q.tmax = 500
q.evaluator = qigacython.KnapsackEvaluator()

q.initialize()

print q.Q

q.run()

print q.best

print q.Q

# qigacython.runcpp()

qigacython.start()

