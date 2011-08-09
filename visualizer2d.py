#!/usr/bin/python

import re,sys
import pylab
import realfuncs

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import cos,sin,pi,sqrt
import matplotlib.pyplot as plt
import numpy as np

from exp import setAttrs

if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.argv += ['realfuncs.Eval2dBin', '--preset', '6']
    args = sys.argv[1:]
    name = args[0].rsplit('.',1)
    Class = getattr(__import__(name[0]), name[1])
    evalobj = Class() # create the evaluator instance
    setAttrs(evalobj, args[1:])

    xmin,xmax,ymin,ymax = (-1.,1.,-1.,1.)
    xmin,xmax,ymin,ymax = (100.,600.,400.,800.)
    step = (xmax - xmin) / 50

    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(xmin, xmax, step)
    Y = np.arange(ymin, ymax, step)
    X, Y = np.meshgrid(X, Y)
    Z = 1. * X
    Z.fill(1)
    for xi in xrange(X.shape[0]):
        for yi in xrange(Y.shape[0]):
            x = X[0,xi]
            y = Y[yi,0]
            val = eval(realfuncs.presets[evalobj.preset]['expr'])
            Z[xi,yi] = val

    f = open('/tmp/log.txt', 'r')
    iter = 0
    state2 = False
    plt.hold(True)
    while True:
        line = f.readline()
        if not line:
            break
        if re.match(r'.*# STAT ', line):
            iter += 1
            print 'generation %d' % iter
            plt.clf()
            ax = Axes3D(fig)
            ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.hot)
            pylab.title('generation %d' % iter)
            plt.xlabel('x')
            plt.ylabel('y')
            ax.set_xlim3d(xmin,xmax)
            ax.set_ylim3d(ymin,ymax)
            state2 = True
            continue
        if state2:
            line = line.strip()
            # match = re.search(r'\(.*,\s+\((.*?),(.*?)\), (.*)\)', line)
            match = re.search(r'\(\[\s*(.*?)\s+(.*?)],.*\s+(.*)\)$', line)
            if match:
                x,y,z=[float(x) for x in match.groups()]
                #print x,y,z
                ax.plot([x],[y],[z],'o',color='red')
            else:
                state2 = False
                #pylab.xlim(xmin,xmax)
                if iter == 60:
                    pass
                    #plt.show()
                plt.savefig('/tmp/aa-%03d.png' % iter)

    sys.exit(0)

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    ax.plot([0],[1],[1], 'o', color='red')
    plt.clf()
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet)
    ax.plot([0],[0],[1], 'o', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('/tmp/bla2.png')
    plt.show()

    sys.exit(0)

logfile = '/tmp/log.txt'
if len(sys.argv) > 1:
    logfile = sys.argv[1]

f = open(logfile, 'r')

while True:
    line = f.readline()
    if not line:
        break
    if line.startswith('## COMMAND:'):
        line = f.readline().strip()
        xmin = float(re.sub(r'.*--xmin (\S+).*', r'\1', line))
        xmax = float(re.sub(r'.*--xmax (\S+).*', r'\1', line))
        expr = re.sub(r'.*--expr (\S+)', r'\1', line)
        break

X = pylab.arange(xmin, xmax, (xmax-xmin)/50)
Y = 1. * X
for i in xrange(X.size):
    x = X[i]
    Y[i] = eval(expr)
pylab.hold(True)
state2 = False
iter = 0
while True:
    line = f.readline()
    if not line:
        break
    if re.match(r'.*# STAT ', line):
        iter += 1
        print 'generation %d' % iter
        pylab.clf()
        pylab.title('generation %d' % iter)
        pylab.plot(X,Y)
        state2 = True
        continue
    if state2:
        if re.match(r'.*\(', line):
            data = re.search(r'\((.*),(.*),(.*)\)', line)
            x,y=[float(x) for x in data.groups()[1:]]
            pylab.plot([x],[y],'o',color='red')
        else:
            state2 = False
            pylab.xlim(xmin,xmax)
            pylab.savefig('/tmp/aa-%03d.png' % iter)



