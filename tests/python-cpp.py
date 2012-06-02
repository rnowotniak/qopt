#!/usr/bin/python

import sys
import numpy as np

import qopt.algorithms.C.algorithms as algorithms
import qopt.benchmarks.CEC2005.cec2005swig as cec2005swig

rqiea = algorithms.rQIEA_ld(5, 2)

# setting and getting parameters of the algorithm
print rqiea.Pc
print rqiea.popsize
rqiea.tmax = 5
rqiea.evaluator = cec2005swig.getfun(1)
print rqiea.popsize

print 'bounds:'
print rqiea.bounds()

print 'seting bounds:'
print rqiea.bounds(np.array([[-50,50]] * rqiea.chromlen, np.double))

# rqiea.bounds = (-100, 100)
print 'bounds:'
print rqiea.bounds()

# running
print
rqiea.run()

print 'best:'
print rqiea.best.fitness

print 'generation 0 state:'
print rqiea.getGeneration(1).P()

print rqiea.getGeneration(3).Q()

#print rqiea.generation


# dalsze rzeczy

#rqiea.bounds = ... ?

print 'koniec'

