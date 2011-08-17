#!/usr/bin/python

import sys
import sys
import pycuda.driver as cuda
import pycuda.gpuarray
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit

dtype = np.double
src = ''.join(open('some-tests.cu').readlines())
mod = SourceModule(src, arch='sm_13')
cuda.memcpy_htod(mod.get_global('a')[0], np.int32(5))

fun = mod.get_function('test')

#jm1 = np.matrix(range(6), dtype).reshape(2,3)
m1 = np.zeros((2,3,4),dtype)
m2 = np.zeros((5,1),dtype)
m3 = np.matrix(range(6), dtype).reshape(2,3)
m4 = np.matrix(range(6), dtype).reshape(2,3)


fun(cuda.InOut(m1), cuda.InOut(m2), cuda.InOut(m3), cuda.InOut(m4), block=(1,1,1), grid=(24,1))

print m1
print m2
print m3
print m4

