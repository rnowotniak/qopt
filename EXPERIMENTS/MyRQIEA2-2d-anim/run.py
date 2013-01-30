#!/usr/bin/python

import sys
import pylab
import numpy as np
import qopt.problems
from qopt.algorithms import MyRQIEA2
import random

random.seed(1)

alg = MyRQIEA2(chromlen = 2, popsize = 10)
alg._initialize()
f = qopt.problems.CEC2013(21)
alg.problem = f

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
from matplotlib import animation


probes = None
distributions = []

def draw_distributions():
    global distributions
    while distributions:
        distributions.pop(0).remove()
    for i in xrange(alg.popsize):
        xy = alg.Q[i][:5][:2] * 100 / (np.pi * 2)
        wh = alg.Q[i][:5][3:5] * 30
        angle = alg.Q[i][:5][2] / (np.pi * 2) * 180
        m = pylab.plot(xy[0], xy[1], 'rs')
        distributions += m
        el = Ellipse(xy=xy, width=wh[0], height=wh[1], angle = angle, alpha = .5)
        distributions.append(el)
        pylab.gca().add_artist(el)

def draw_probes():
    global probes
    if probes is not None:
        probes.pop(0).remove()
    alg._observe();
    probes = pylab.plot(alg.P[:,0], alg.P[:,1], 'go')

pylab.xlim(-50,50);
pylab.ylim(-50,50);

def update_plot(i):
    print i
    alg.generation()
    print alg.best
    draw_distributions()
    draw_probes()

anim = animation.FuncAnimation(pylab.gcf(), update_plot,repeat=True,frames=300, interval=1000/30,blit=True)

anim.save('/tmp/MyRQIEA.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

#pylab.show()

