#!/usr/bin/python

import sys
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit

print cuda.Device.count()

src = ''.join(open('kern1.cu').readlines())

f = SourceModule(src, arch='sm_13').get_function('f')

a = np.random.randn(4)
print a.astype(np.double)

print f
print f(cuda.Out(a), block=(4,1,1),grid=(1,1), time_kernel=True)

print a.astype(np.double)



