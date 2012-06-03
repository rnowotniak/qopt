#!/usr/bin/python

import os
import os.path
import sys
import ctypes
from math import pi

LIBS = {}

for fnum in xrange(1, 26):
    LIBS['f%d' % fnum] = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + '/libf%s.so' % fnum)
    LIBS['f%d' % fnum].evaluate.restype = ctypes.c_longdouble

_FUNCTIONS_DATA =  (
        #     bounds     opt           plot_bounds       plot_density
        ( (-100, 100), -450, ( (-100, 100),  (-100, 100)), 30 ), # f1       
        ( (-100, 100), -450, ( (-100, 100),  (-100, 100)), 30 ), # f2
        ( (-100, 100), -450, ( (-100, 100),  (-100, 100)), 30 ), # f3
        ( (-100, 100), -450, (    (0, 100),    (-100, 0)), 30 ), # f4
        ( (-100, 100), -310, ( (-200, 200),  (-200, 200)), 30 ), # f5
        ( (-100, 100),  390, (    (78, 82),   (-52, -47)), 30 ), # f6
        (   (-0, 600), -180, ((-340, -250),    (-100, 0)), 30 ), # f7
        (   (-32, 32), -140, (   (-40, 40),    (-40, 40)), 70 ), # f8
        (     (-5, 5), -330, (     (-5, 5),      (-5, 5)), 50 ), # f9
        (     (-5, 5), -330, (     (-5, 5),      (-5, 5)), 50 ), # f10
        (   (-.5, .5),   90, (   (-.5, .5),    (-.5, .5)), 60 ), # f11
        (   (-pi, pi), -460, (     (-4, 4),      (-4, 4)), 50 ), # f12
        (     (-3, 1), -130, (    (-2, -1),     (-2, -1)), 50 ), # f13
        ( (-100, 100), -300, (  (-90, -50),     (-40, 0)), 70 ), # f14
        (     (-5, 5),  120, (     (-5, 5),      (-5, 5)), 60 ), # f15
        (     (-5, 5),  120, (     (-5, 5),      (-5, 5)), 60 ), # f16
        (     (-5, 5),  120, (     (-5, 5),      (-5, 5)), 60 ), # f17
        (     (-5, 5),   10, (     (-5, 5),      (-5, 5)), 60 ), # f18
        (     (-5, 5),   10, (     (-5, 5),      (-5, 5)), 60 ), # f19
        (     (-5, 5),   10, (     (-5, 5),      (-5, 5)), 60 ), # f20
        (     (-5, 5),  360, (     (-5, 5),      (-5, 5)), 60 ), # f21
        (     (-5, 5),  360, (     (-5, 5),      (-5, 5)), 60 ), # f22
        (     (-5, 5),  360, (     (-5, 5),      (-5, 5)), 60 ), # f23
        (     (-5, 5),  260, (     (-5, 5),      (-5, 5)), 60 ), # f24
        (      (2, 5),  260, (     (-5, 5),      (-5, 5)), 60 )  # f25
        )

def fgenerator(num):
    def fun(arg):
        return f(num, arg)
    return fun

FUNCTIONS = {}
for fnum in xrange(1, 26):
    FUNCTIONS['f%d' % fnum] = {
            'fun'          : fgenerator(fnum),
            'cfun'         : LIBS['f%d' % fnum].evaluate,
            'bounds'       : _FUNCTIONS_DATA[fnum - 1][0],
            'opt'          : _FUNCTIONS_DATA[fnum - 1][1],
            'plot_bounds'  : _FUNCTIONS_DATA[fnum - 1][2],
            'plot_density' : _FUNCTIONS_DATA[fnum - 1][3]
            }

# pliki libf%d.so to maja byc biblioteki zawierajace pelen benchmark skompilowany dla funkcji testowej %d

def f(num, x):
    fnum = 'f%d' % num
#    if not LIBS.has_key(fnum):
#        lib=ctypes.CDLL('./lib%s.so' % fnum)
#        LIBS[fnum] = lib
#        LIBS[fnum].evaluate.restype = ctypes.c_longdouble
    LIBS[fnum].evaluate.argtypes = [ctypes.c_longdouble * len(x), ctypes.c_int]
    arr = ctypes.c_longdouble * len(x)
    x = arr(*x)
    return LIBS[fnum].evaluate(x, len(x))

## (previously the benchmark functions initialization was performed here in Python directly)
#   if not LIBS.has_key(fnum):
#       # load the library with proper benchmark function
#       lib=ctypes.CDLL('./lib%s.so' % fnum) # , ctypes.RTLD_GLOBAL)
#       # store the loaded library in LIBS
#       LIBS[fnum] = [lib, None, None]
#   else:
#       # library for this function is already loaded in LIBS
#       lib = LIBS[fnum][0]
#   if LIBS[fnum][1] != len(x):
#       # initialize the library for the given dimension (len(x))
#       nreal = len(x)
#       if num >= 15:
#           nfunc = 10
#       else:
#           nfunc = 2
#       LIBS[fnum][1] = nreal
#       LIBS[fnum][2] = nfunc
#       ctypes.c_int.in_dll(lib, 'nreal').value = nreal
#       ctypes.c_int.in_dll(lib, 'nfunc').value = nfunc
#       lib.randomize()
#       lib.initrandomnormaldeviate()
#       lib.allocate_memory()
#       lib.initialize()
#       print
#       if num >= 15:
#           lib.calc_benchmark_norm()
#       lib.calc_benchmark_func.restype = ctypes.c_longdouble
#       lib.calc_benchmark_func.argtypes = [ctypes.c_longdouble * nreal]
#   arr = ctypes.c_longdouble * len(x)
#   x = arr(*x)
#   return lib.calc_benchmark_func(x)

for fnum in xrange(1, 26):
    globals()['f%d' % fnum] = fgenerator(fnum)


if __name__ == '__main__':
    # print f3((-2, 5))
    # print f1((-39.311, 58.899))
    # print f15((3.325, -1.2835))
    # f3((-2.1, 5.5))
    # f3((-3.2201300e+001, 6.4977600e+001 ))
    # f1((-39, 58, -46, -74))
    # print f16((-5,-4))
    # sys.exit(0)

    import sys
    import numpy as np

    import matplotlib
    matplotlib.use('cairo')
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter, ScalarFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    print 'Generating 2D plots...'
    for fnum in xrange(1, 26):
        print '%d ' % fnum
        sys.stdout.flush()

        X = np.linspace(*(FUNCTIONS['f%d'%fnum]['plot_bounds'][0] + (FUNCTIONS['f%d'%fnum]['plot_density'],)))
        Y = np.linspace(*(FUNCTIONS['f%d'%fnum]['plot_bounds'][1] + (FUNCTIONS['f%d'%fnum]['plot_density'],)))

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        X, Y = np.meshgrid(X, Y)
        Z = np.zeros(X.shape)
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                Z[i,j] = f(fnum, (X[0,j], Y[i,0]))

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0.5, antialiased=False)
        cset = ax.contour(X, Y, Z, zdir='z', offset = 0)

        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        plt.title('$F_{%d}(x,y)$'%fnum)

        plt.savefig('/tmp/f_%d.png'%fnum, bbox_inches='tight')

