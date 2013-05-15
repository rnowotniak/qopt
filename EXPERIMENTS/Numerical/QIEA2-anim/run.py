#!/usr/bin/python

import sys
import pylab
import numpy as np
import qopt.problems
import qopt.algorithms
import random
from matplotlib.patches import Ellipse

f = qopt.problems.CEC2013(1)
X = np.linspace(-100, 100,30)
Y = np.linspace(-100, 100,30)

X,Y = np.meshgrid(X,Y)
Z = f.evaluate((X,Y))

surf = pylab.contourf(X,Y, Z)#, cmap='hot')
opt = f.optimum[:2]
pylab.plot(opt[0], opt[1], 'ro')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.colorbar(surf, format = '%g')
pylab.xlim(-100,100);
pylab.ylim(-100,100);

random.seed(3)

toRemove = []

global NR
NR = 0

class QIEA2(qopt.algorithms.QIEA2):
    def generation(self):
        global toRemove, NR
        #print self.Q[0][0]
        super(QIEA2, self).generation()
        for i in xrange(alg.popsize):
            xy = self.Q[i][0:2]
            #print xy
            wh = self.Q[i][:5][3:5] # * 30
            angle = self.Q[i][:5][2] / (np.pi * 2) * 180
            el = Ellipse(xy=xy, width=wh[0], height=wh[1], angle = angle, alpha = .5)
            pylab.gca().add_artist(el)
            m = pylab.plot(xy[0], xy[1], 'rs')
            toRemove.append(el)
            toRemove += m
        for i in xrange(alg.K):
            xy = self.P[i][0:2]
            m = pylab.plot(xy[0], xy[1], 'ys')
            toRemove += m
            print xy
        pylab.savefig('/tmp/blaa-%03d.png' % NR)
        while toRemove:
            toRemove.pop(0).remove()

        NR += 1
        #sys.exit(0)
        #print self.bestval


alg = QIEA2(chromlen = 2, popsize = 10)
alg.problem = f
alg.XI = 0
alg.tmax = 1000
alg.run()
print alg.bestval

sys.exit(0)
###########



X = np.linspace(-100, 100,30)
Y = np.linspace(-100, 100,30)

X,Y = np.meshgrid(X,Y)
Z = f.evaluate((X,Y))

surf = pylab.contourf(X,Y, Z)#, cmap='hot')
opt = f.optimum[:2]
pylab.plot(opt[0], opt[1], 'ro')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.colorbar(surf, format = '%g')
pylab.xlim(-100,100);
pylab.ylim(-100,100);

pylab.savefig('/tmp/blaa.png')

sys.exit(0)






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


def update_plot(i):
    print i
    alg.generation()
    print alg.best
    draw_distributions()
    draw_probes()

anim = animation.FuncAnimation(pylab.gcf(), update_plot,repeat=True,frames=300, interval=1000/30,blit=False)
anim.save('/tmp/MyRQIEA.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

#pylab.show()

