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
        ( (-100, 100), -450, ( (-100, 100),  (-100, 100)), 30,          1e-6 ), # f1       
        ( (-100, 100), -450, ( (-100, 100),  (-100, 100)), 30,          1e-6 ), # f2
        ( (-100, 100), -450, ( (-100, 100),  (-100, 100)), 30,          1e-6 ), # f3
        ( (-100, 100), -450, (    (0, 100),    (-100, 0)), 30,          1e-6 ), # f4
        ( (-100, 100), -310, ( (-200, 200),  (-200, 200)), 30,          1e-6 ), # f5
        ( (-100, 100),  390, (    (78, 82),   (-52, -47)), 30,          1e-2 ), # f6
        (   (-0, 600), -180, ((-340, -250),    (-100, 0)), 30,          1e-2 ), # f7
        (   (-32, 32), -140, (   (-40, 40),    (-40, 40)), 70,          1e-2 ), # f8
        (     (-5, 5), -330, (     (-5, 5),      (-5, 5)), 50,          1e-2 ), # f9
        (     (-5, 5), -330, (     (-5, 5),      (-5, 5)), 50,          1e-2 ), # f10
        (   (-.5, .5),   90, (   (-.5, .5),    (-.5, .5)), 60,          1e-2 ), # f11
        (   (-pi, pi), -460, (     (-4, 4),      (-4, 4)), 50,          1e-2 ), # f12
        (     (-3, 1), -130, (    (-2, -1),     (-2, -1)), 50,          1e-2 ), # f13
        ( (-100, 100), -300, (  (-90, -50),     (-40, 0)), 70,          1e-2 ), # f14
        (     (-5, 5),  120, (     (-5, 5),      (-5, 5)), 60,          1e-2 ), # f15
        (     (-5, 5),  120, (     (-5, 5),      (-5, 5)), 60,          1e-2 ), # f16
        (     (-5, 5),  120, (     (-5, 5),      (-5, 5)), 60,          1e-1 ), # f17
        (     (-5, 5),   10, (     (-5, 5),      (-5, 5)), 60,          1e-1 ), # f18
        (     (-5, 5),   10, (     (-5, 5),      (-5, 5)), 60,          1e-1 ), # f19
        (     (-5, 5),   10, (  (  2, 21),     (-32, -12)), 60,          1e-1 ), # f20
        (     (-5, 5),  360, (     (-50, 50),      (-50, 50)), 60,          1e-1 ), # f21
        (     (-5, 5),  360, (     (-100, 100),      (-100, 100)), 60,          1e-1 ), # f22
        (     (-5, 5),  360, (     (-80, 80),      (-80, 80)), 60,          1e-1 ), # f23
        (     (-5, 5),  260, (     (-100, 100),      (-100, 100)), 60,          1e-1 ), # f24
        (      (2, 5),  260, (     (-100, 100),      (-100, 100)), 60,          1e-1 ), # f25
        (      (2, 5),  260, (     (-100, 100),      (-100, 100)), 60,          1e-1 ), # f26
        (      (2, 5),  260, (     (-100, 100),      (-100, 100)), 60,          1e-1 ), # f27
        (      (2, 5),  260, (     (-80, 80),      (-80, 80)), 60,          1e-1 )  # f25
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

    print 'Generating 2D plots...'
    for fnum in xrange(21, 29):
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
        #cset = ax.contour(X, Y, Z, zdir='z', offset = 0)

        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.title('$f_{%d}(x,y)$'%fnum)

        plt.savefig('/tmp/f_%d.png'%fnum, bbox_inches='tight')

