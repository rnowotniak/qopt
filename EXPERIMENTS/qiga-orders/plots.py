#!/usr/bin/python
# encoding: utf-8

import sys

import pylab
import numpy as np

def lambda_(x, r):
    return 1. * 2**r * np.log(2**x) / np.log(2**r) / 2**x

def lambdaO(x, r):
    return 1. * 2**(2*r) * np.log(2**x) / np.log(2**r) / 2**(2*x)

X = np.linspace(5,12,200)
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
pylab.plot(X,R2, label='$r=1,r=2$')
pylab.plot(X,R3, label='$r=3$')
pylab.plot(X,R4, label='$r=4$')
pylab.plot(X,R5, label='$r=5$')
pylab.legend(loc = 'upper right')
pylab.xlabel('rozmiar zadania $N$')
pylab.ylabel('Współczynnik kwantowości $\lambda$')
pylab.title(u'Współczynnik kwantowości $\lambda$ dla różnych wartości $r$ i $N$')
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

pylab.plot(X,R1, label='$r=1$')
pylab.plot(X,R2, label='$r=2$')
pylab.plot(X,R3, label='$r=3$')
pylab.plot(X,R4, label='$r=4$')
pylab.plot(X,R5, label='$r=5$')
pylab.legend(loc = 'upper right')
pylab.xlabel('rozmiar zadania $N$')
pylab.ylabel('Współczynnik kwantowości $\lambda_O$')
pylab.title(u'Współczynnik kwantowości $\lambda_O$ dla różnych wartości $r$ i $N$')
pylab.savefig('/tmp/orders-operators.pdf', bbox_inches = 'tight')

