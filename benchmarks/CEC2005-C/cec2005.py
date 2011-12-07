#!/usr/bin/python

import os
import os.path
import sys
import ctypes

LIBS = {}

os.chdir(os.path.dirname(os.path.realpath(__file__)))

# pliki libf%d.so to maja byc biblioteki zawierajace pelen benchmark skompilowany dla funkcji testowej %d

def f(num, x):
    fnum = 'f%d' % num
    if not LIBS.has_key(fnum):
        # load the library with proper benchmark function
        lib=ctypes.CDLL('./lib%s.so' % fnum) # , ctypes.RTLD_GLOBAL)
        # store the loaded library in LIBS
        LIBS[fnum] = [lib, None, None]
    else:
        # library for this function is already loaded in LIBS
        lib = LIBS[fnum][0]
    if LIBS[fnum][1] != len(x):
        # initialize the library for the given dimension (len(x))
        nreal = len(x)
        if num >= 15:
            nfunc = 10
        else:
            nfunc = 2
        LIBS[fnum][1] = nreal
        LIBS[fnum][2] = nfunc
        ctypes.c_int.in_dll(lib, 'nreal').value = nreal
        ctypes.c_int.in_dll(lib, 'nfunc').value = nfunc
        lib.randomize()
        lib.initrandomnormaldeviate()
        lib.allocate_memory()
        lib.initialize()
        print
        if num >= 15:
            lib.calc_benchmark_norm()
        lib.calc_benchmark_func.restype = ctypes.c_longdouble
        lib.calc_benchmark_func.argtypes = [ctypes.c_longdouble * nreal]
    arr = ctypes.c_longdouble * len(x)
    x = arr(*x)
    return lib.calc_benchmark_func(x)


def fgenerator(num):
    def fun(arg):
        return f(num, arg)
    return fun

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
    matplotlib.use('pdf')
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter, ScalarFormatter
    import matplotlib.pyplot as plt
    import numpy as np

    print 'Generating 2D plots...'
    for fnum in xrange(15, 26):
        print 'f_%d' % fnum
        sys.stdout.flush()
        if fnum > 4:
            X, Y = np.linspace(-100, 100, 30), np.linspace(-100, 100, 30)
        if fnum == 4:
            X, Y = np.linspace(0, 100, 30), np.linspace(-100, 0, 30)
        elif fnum == 5:
            X, Y = np.linspace(-200, 200, 30), np.linspace(-200, 200, 30)
        elif fnum == 6:
            X, Y = np.linspace(78, 82, 30), np.linspace(-47, -52, 30)
        elif fnum == 7:
            X, Y = np.linspace(-350, -250, 30), np.linspace(-100, 0, 30)
        elif fnum == 8:
            X, Y = np.linspace(-40, 40, 70), np.linspace(-40, 40, 70)
        elif fnum == 9:
            X, Y = np.linspace(-5, 5, 50), np.linspace(-5, 5, 50)
        elif fnum == 10:
            X, Y = np.linspace(-5, 5, 50), np.linspace(-5, 5, 50)
        elif fnum == 11:
            X, Y = np.linspace(-.5, .5, 60), np.linspace(-.5, .5, 60)
        elif fnum == 12:
            X, Y = np.linspace(-4, 4, 50), np.linspace(-4, 4, 50)
        elif fnum == 13:
            X, Y = np.linspace(-2, -1, 50), np.linspace(-2, -1, 50)
        elif fnum == 14:
            X, Y = np.linspace(-90, -50, 70), np.linspace(-40, 0, 70)
        elif fnum >= 15:
            X, Y = np.linspace(-5, 5, 60), np.linspace(-5, 5, 60)
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

        plt.savefig('/tmp/f_%d.pdf'%fnum, bbox_inches='tight')
        #plt.show()
