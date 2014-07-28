#!/usr/bin/python
# encoding: utf-8

import sys

import pylab
import numpy as np

def lambda_(x, r):
    return 1. * 2**r * np.log(2**x) / np.log(2**r) / 2**x

def lambdaO(x, r):
    return 1. * 2**(2*r) * np.log(2**(x)) / np.log(2**r) / 2**(2*x)

X = np.arange(5,12,1)
R1 = []
R2 = []
R3 = []
R4 = []
R5 = []
for x in X:
    R1.append(lambda_(x, 1))
    R2.append(lambda_(x, 2))
    R3.append(lambda_(x, 3))
    R4.append(lambda_(x, 4))
    R5.append(lambda_(x, 5))

pylab.plot(X,R1)
pylab.plot(X,R2, 'bo-', label='$r=1,r=2$', markersize=10)
pylab.plot(X,R3, 'rx-', label='$r=3$', markersize=12)
pylab.plot(X,R4, 'c*-', label='$r=4$', markersize=12)
pylab.plot(X,R5, 'mD-', label='$r=5$')
pylab.legend(loc = 'upper right')
pylab.xlabel('Problem size $N$')
pylab.ylabel('Quantum factor $\lambda$')
pylab.title(u'Quantum factor $\lambda$ for different $r$ and $N$')
pylab.savefig('/tmp/orders.pdf', bbox_inches = 'tight')

pylab.figure()

R1 = []
R2 = []
R3 = []
R4 = []
R5 = []
for x in X:
    R1.append(lambdaO(x, 1))
    R2.append(lambdaO(x, 2))
    R3.append(lambdaO(x, 3))
    R4.append(lambdaO(x, 4))
    R5.append(lambdaO(x, 5))

pylab.plot(X,R1, 'bo-', label='$r=1$')
pylab.plot(X,R2, 'gs-.', label='$r=2$')
pylab.plot(X,R3, 'rx-', label='$r=3$')
pylab.plot(X,R4, 'c*-', label='$r=4$')
pylab.plot(X,R5, 'mD-', label='$r=5$')
pylab.legend(loc = 'upper right')
pylab.xlabel('Problem size $N$')
pylab.ylabel('Quantum operators factor $\lambda_O$')
pylab.title(u'Quantum operators factor $\lambda_O$ for different $r$ i $N$')
pylab.savefig('/tmp/orders-operators.pdf', bbox_inches = 'tight')

