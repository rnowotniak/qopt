#!/usr/bin/python

import os
import sys
import ctypes

LIBS = {}

# pliki libf%d.so to maja byc biblioteki zawierajace pelen benchmark skompilowany dla funkcji testowej %d

def f(num, x):
    fnum = 'f%d' % num
    if not LIBS.has_key(fnum):
        lib=ctypes.CDLL('./lib%s.so' % fnum) # , ctypes.RTLD_GLOBAL)
        LIBS[fnum] = [lib, None, None]
    else:
        lib = LIBS[fnum][0]
    if LIBS[fnum][1] != len(x):
        nreal = len(x)
        nfunc = 2
        LIBS[fnum][1] = nreal
        LIBS[fnum][2] = nfunc
        ctypes.c_int.in_dll(lib, 'nreal').value = nreal
        ctypes.c_int.in_dll(lib, 'nfunc').value = nfunc
        lib.randomize()
        lib.initrandomnormaldeviate()
        lib.allocate_memory()
        lib.initialize()
        lib.calc_benchmark_func.restype = ctypes.c_longdouble
        lib.calc_benchmark_func.argtypes = [ctypes.c_longdouble * nreal]
    arr = ctypes.c_longdouble * len(x)
    x = arr(*x)
    print
    print lib.calc_benchmark_func(x)


def fgenerator(num):
    def fun(arg):
        f(num, arg)
    return fun

for fnum in xrange(1, 26):
    globals()['f%d' % fnum] = fgenerator(fnum)


if __name__ == '__main__':
    f3((-2, 5))
    f3((-2.1, 5.5))
    f3((-3.2201300e+001, 6.4977600e+001 ))
    f1((-39, 58, -46, -74))

