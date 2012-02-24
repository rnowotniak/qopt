#!/usr/bin/python

import sys
import ctypes


lib = ctypes.CDLL('./a.out')
ev = ctypes.CDLL('./evaluator.so')

print dir(lib)

lib.foo(ev.sum)
ctypes.c_double.in_dll(lib, 'somevar').value = 66;

lib.init();
ctypes.c_double.in_dll(lib, 'tmax').value = 2;
lib.run();


