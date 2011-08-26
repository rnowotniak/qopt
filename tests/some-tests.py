#!/usr/bin/python

import sys
import pycuda.driver as cuda
import pycuda.gpuarray
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit
import time

dtype = np.double
src = ''.join(open('some-tests.cu').readlines())
mod = SourceModule(src, arch='sm_13', no_extern_c = True)

mod.get_global('str')

fun = mod.get_function('test')

#jm1 = np.matrix(range(6), dtype).reshape(2,3)
m1 = np.zeros((2,3),dtype)

fun(cuda.InOut(m1), np.uint32(time.time()), block=(10,1,1), grid=(1,1))

print m1
