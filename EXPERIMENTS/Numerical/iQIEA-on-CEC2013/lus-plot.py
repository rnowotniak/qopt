#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
import pylab


f=open('results50-sorted.txt')

X=[]
Y=[]

n=1
while True:
    line = f.readline()
    if not line:
        break
    val = float(line.split()[2])
    print n, val
    X.append(n)
    Y.append(val)
    n += 1

pylab.xlabel(u'Numer iteracji $t$ algorytmu LUS')
pylab.ylabel(u'Wartość metadopasowania $\\tilde{f}$ algorytmu iQIEA')
pylab.xticks(())

pylab.plot(X,Y)
pylab.savefig('/tmp/lus-plot.pdf', bbox_inches='tight')


