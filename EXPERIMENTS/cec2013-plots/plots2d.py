#!/usr/bin/python

import os
import os.path
import sys
import ctypes
from math import pi

LIBS = {}

# for fnum in xrange(1, 26):
#     LIBS['f%d' % fnum] = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + '/libf%s.so' % fnum)
#     LIBS['f%d' % fnum].evaluate.restype = ctypes.c_longdouble

_FUNCTIONS_DATA =  (
        #     bounds     opt           plot_bounds       plot_density   accuracy
        ( (-100, 100), -1400, ( (-100, 100),  (-100, 100)), 30,          1e-6 ), # f1       
        ( (-100, 100), -1300, ( (-100, 100),  (-100, 100)), 30,          1e-6 ), # f2
        ( (-100, 100), -1200, ( (-23, -20),  (10, 13)), 50,          1e-6 ), # f3
        ( (-100, 100), -1100, (    (-100, 100),    (-100, 100)), 30,          1e-6 ), # f4
        ( (-100, 100), -1000, ( (-100, 100),  (-100, 100)), 30,          1e-6 ), # f5
        ( (-100, 100), -900, (    (-70, 20),   (-30, 60)), 30,          1e-2 ), # f6
        (   (-0, 600), -800, ((-26, -17),    (7, 16)), 30,          1e-2 ), # f7
        (   (-32, 32), -700, (   (-40, 40),    (-40, 40)), 70,          1e-2 ), # f8
        (     (-5, 5), -600, (    (-50, 50),      (-50, 50)), 50,          1e-2 ), # f9
        (     (-5, 5), -500, (     (-26, -17),      (7, 16)), 50,          1e-2 ), # f10
        (   (-.5, .5), -400, (   (-40, -5),    (-5, 30)), 60,          1e-2 ), # f11
        (   (-pi, pi), -300, (     (-40, -5),      (0, 30)), 50,          1e-2 ), # f12
        (     (-3, 1), -200, (    (-40, -5),     (-3, 30)), 50,          1e-2 ), # f13
        ( (-100, 100), -100, (  (-100, 100),     (-100, 100)), 70,          1e-2 ), # f14
        (     (-5, 5), 100, (     (-100, 100),      (-100, 100)), 60,          1e-2 ), # f15
        (     (-5, 5), 200, (     (-30, -15),      (5, 20)), 60,          1e-2 ), # f16
        (     (-5, 5), 300, (     (-100, 100),      (-100, 100)), 60,          1e-1 ), # f17
        (     (-5, 5), 400, (     (-100, 100),      (-100, 100)), 60,          1e-1 ), # f18
        (     (-5, 5), 500, (     (-30, -15),      (5, 20)), 60,          1e-1 ), # f19
        (     (-5, 5), 600, (  (  -32, -12),     (3, 20)), 60,          1e-1 ), # f20
        (     (-5, 5), 700, (     (-50, 50),      (-50, 50)), 60,          1e-1 ), # f21
        (     (-5, 5), 800, (     (-100, 100),      (-100, 100)), 60,          1e-1 ), # f22
        (     (-5, 5), 900, (     (-80, 80),      (-80, 80)), 70,          1e-1 ), # f23
        (     (-5, 5), 1000, (     (-80, 80),      (-70, 70)), 70,          1e-1 ), # f24
        (      (2, 5), 1100, (     (-100, 100),      (-100, 100)), 70,          1e-1 ), # f25
        (      (2, 5), 1200, (     (-60, 18),      (-28, 40)), 70,          1e-1 ), # f26
        (      (2, 5), 1300, (     (-100, 100),      (-100, 100)), 40,          1e-1 ), # f27
        (      (2, 5), 1400, (     (-80, 80),      (-80, 80)), 70,          1e-1 )  # f28
        )

def fgenerator(num):
    def fun(arg):
        return f(num, arg)
    return fun

FUNCTIONS = {}
for fnum in xrange(1, 29):
    FUNCTIONS['f%d' % fnum] = {
            'bounds'       : _FUNCTIONS_DATA[fnum - 1][0],
            'opt'          : _FUNCTIONS_DATA[fnum - 1][1],
            'plot_bounds'  : _FUNCTIONS_DATA[fnum - 1][2],
            'plot_density' : _FUNCTIONS_DATA[fnum - 1][3],
            'accuracy'     : _FUNCTIONS_DATA[fnum - 1][4]
            }

# pliki libf%d.so to maja byc biblioteki zawierajace pelen benchmark skompilowany dla funkcji testowej %d

for fnum in xrange(1, 26):
    globals()['f%d' % fnum] = fgenerator(fnum)

if __name__ == '__main__':

    import sys
    import numpy as np

    import matplotlib
    matplotlib.use('cairo')
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter, ScalarFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    import qopt.problems

    fnums = [int(f) for f in sys.argv[1:]]

    print 'Generating 2D plots...'
    for fnum in fnums:
        print '%d ' % fnum
        sys.stdout.flush()

        X = np.linspace(*(FUNCTIONS['f%d'%fnum]['plot_bounds'][0] + (FUNCTIONS['f%d'%fnum]['plot_density'],)))
        Y = np.linspace(*(FUNCTIONS['f%d'%fnum]['plot_bounds'][1] + (FUNCTIONS['f%d'%fnum]['plot_density'],)))

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(X.shape)
        f = qopt.problems.CEC2013(fnum)
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                Z[i,j] = f.evaluate((X[0,j], Y[i,0]))

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.5, antialiased=False)
        cset = ax.contour(X, Y, Z, zdir='z', offset = FUNCTIONS['f%d'%fnum]['opt'])

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.title('$f_{%d}(x,y)$'%fnum)

        plt.savefig('/tmp/f_%02d.png'%fnum, bbox_inches='tight')

