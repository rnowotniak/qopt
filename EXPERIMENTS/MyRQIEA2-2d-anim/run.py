#!/usr/bin/python

import sys
import pylab
import numpy as np
import qopt.problems
from qopt.algorithms import MyRQIEA2

alg = MyRQIEA2(chromlen = 2, popsize = 10)
alg._initialize()

# alg._observe()
# print alg.P
# alg._observe()
# print alg.P

#sys.exit(0)

f = qopt.problems.CEC2013(21)

X = np.linspace(-50, 50,30)
Y = np.linspace(-50, 50,30)

X,Y = np.meshgrid(X,Y)
Z = f.evaluate((X,Y))

surf = pylab.contourf(X,Y, Z)
opt = f.optimum[:2]
pylab.plot(opt[0], opt[1], 'ro')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.colorbar(surf, format = '%g')


from matplotlib.patches import Ellipse

for i in xrange(alg.popsize):
    xy = alg.Q[i][:5][:2] * 100 / (np.pi * 2)
    wh = alg.Q[i][:5][3:5] * 30
    angle = alg.Q[i][:5][2] / (np.pi * 2) * 180
    pylab.plot(xy[0], xy[1], 'rs')
    el = Ellipse(xy=xy, width=wh[0], height=wh[1], angle = angle, alpha = .5)
    pylab.gca().add_artist(el)

for probes in xrange(100):
    alg._observe();
    for i in xrange(alg.popsize):
        p = alg.P[i]
        pylab.plot(p[0], p[1], 'go')


pylab.xlim(-50,50);
pylab.ylim(-50,50);

pylab.show()

