#!/usr/bin/python
# encoding: utf-8

import sys

import pylab
import numpy as np
import math

def lambda_(x, r):
    return 1. * 2**r * (1.*x/r) / 2**x

def lambda_2(x, r):
    return (1. * 2**r * math.floor(1.*x/r) + 2**(x % r)  ) / 2**x

def lambdaO(x, r):
    return 1. * 2**(2*r) * (1.*x/r) / 2**(2*x)

X  = np.arange(4,12,1)
X5 = np.arange(5,12,1)
#R1 = []
#R2 = []
#R3 = []
#R4 = []
#R5 = []
#for x in X:
#    R1.append(lambda_(x, 1))
#    R2.append(lambda_(x, 2))
#    R3.append(lambda_(x, 3))
#    R4.append(lambda_(x, 4))
#    R5.append(lambda_(x, 5))

pylab.plot( X,map(lambda x: lambda_(x,1), X), '--') # , label='$r=1$')
pylab.plot( X,map(lambda x: lambda_(x,2), X), '--') # , label='$r=2$')  
pylab.plot( X,map(lambda x: lambda_(x,3), X), '--') # , label='$r=3$')
pylab.plot( X,map(lambda x: lambda_(x,4), X), '--') # , label='$r=4$')
pylab.plot(X5,map(lambda x: lambda_(x,5),X5), '--') # , label='$r=5$')
# o
# x
# *
# D

#pylab.plot(X, map(lambda x: lambda_(x,1), X), 'ko-')
#pylab.plot(X, map(lambda x: lambda_(x,2), X), 'ko-')
#pylab.plot(X, map(lambda x: lambda_(x,3), X), 'ko-')
#pylab.plot(X, map(lambda x: lambda_(x,4), X), 'ko-')
#pylab.plot(X, map(lambda x: lambda_(x,5), X), 'ko-')

pylab.plot([4,5,6,7,8,9,10], map(lambda x: lambda_(x,2), [4,5,6,7,8,9,10]), 'wo', markersize=15, label='$r=1$')  # R1
pylab.plot([4,6,8,10], map(lambda x: lambda_(x,2), [4,6,8,10]),             'bo', markersize= 8, label='$r=2$')  # R2
pylab.plot([6,9],    map(lambda x: lambda_(x,3), [6,9]),                    'r^', markersize=12, label='$r=3$')  # R3
pylab.plot([4,8],      map(lambda x: lambda_(x,4), [4,8]),                  'cD', markersize=10, label='$r=4$')  # R4
pylab.plot([5,10],   map(lambda x: lambda_(x,5), [5,10]),                   'm*', markersize=19, label='$r=5$')  # R5

pylab.legend(loc = 'upper right')
pylab.xlabel('rozmiar zadania $N$')
pylab.ylabel(u'Współczynnik kwantowości $\lambda$')
pylab.title(u'Współczynnik kwantowości $\lambda$ dla różnych wartości $r$ i $N$')
pylab.xlim(4, 11)
pylab.ylim(0, 1.05)


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


pylab.plot( X,map(lambda x: lambdaO(x,1), X), '--') # , label='$r=1$')
pylab.plot( X,map(lambda x: lambdaO(x,2), X), '--') # , label='$r=2$')  
pylab.plot( X,map(lambda x: lambdaO(x,3), X), '--') # , label='$r=3$')
pylab.plot( X,map(lambda x: lambdaO(x,4), X), '--') # , label='$r=4$')
pylab.plot(X5,map(lambda x: lambdaO(x,5),X5), '--') # , label='$r=5$')

pylab.plot([4,5,6,7,8,9,10], map(lambda x: lambdaO(x,2), [4,5,6,7,8,9,10]), 'wo', markersize=15, label='$r=1$')  # R1
pylab.plot([4,6,8,10], map(lambda x: lambdaO(x,2), [4,6,8,10]),             'bo', markersize= 8, label='$r=2$')  # R2
pylab.plot([6,9],    map(lambda x: lambdaO(x,3), [6,9]),                    'r^', markersize=12, label='$r=3$')  # R3
pylab.plot([4,8],      map(lambda x: lambdaO(x,4), [4,8]),                  'cD', markersize=10, label='$r=4$')  # R4
pylab.plot([5,10],   map(lambda x: lambdaO(x,5), [5,10]),                   'm*', markersize=19, label='$r=5$')  # R5

# pylab.plot(X,R1, 'bo--', label='$r=1$')
# pylab.plot(X,R2, 'gs--', label='$r=2$')
# pylab.plot(X,R3, 'rx--', label='$r=3$')
# pylab.plot(X,R4, 'c*--', label='$r=4$')
# pylab.plot(X,R5, 'mD--', label='$r=5$')

pylab.xlim(4, 11)
pylab.ylim(0, 1.05)

pylab.legend(loc = 'upper right')
pylab.xlabel('rozmiar zadania $N$')
pylab.ylabel(u'Współczynnik kwantowości $\lambda_O$')
pylab.title(u'Współczynnik kwantowości $\lambda_O$ dla różnych wartości $r$ i $N$')
pylab.savefig('/tmp/orders-operators.pdf', bbox_inches = 'tight')

