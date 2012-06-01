#!/usr/bin/python

import sys
import ctypes


lib = ctypes.CDLL('./a.out')

ctypes.c_double.in_dll(lib, 'somevar').value = 13;

print ctypes.c_double.in_dll(lib, 'somevar').value

ev = ctypes.CDLL('./evaluator.so')

print dir(lib)

#FTYPE = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int)
print ev.sum
print ev.sum
ctypes.c_void_p.in_dll(lib, 'evaluator').value = ctypes.cast(ev.sum, ctypes.c_void_p).value
print 'aa'
print ctypes.cast(ev.sum, ctypes.c_void_p).value
lib.foo(ev.sum)

sys.exit(0)
ctypes.c_void_p.in_dll(lib, 'evaluator').value = ctypes.POINTER(ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.c_int)(ev.sum))

lib.foo(ev.sum)
ctypes.c_double.in_dll(lib, 'somevar').value = 66;

lib.init();
ctypes.c_double.in_dll(lib, 'tmax').value = 2;
lib.run();

print 'koniec'

