#!/usr/bin/python
import sys
import pycuda.driver as cuda
import pycuda.gpuarray
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit

nreal = 2;
nfunc = 3;

dtype = np.double

print 'devices:', cuda.Device.count()

# memory allocation
GPU_ptrs = []
MEM = cuda.mem_alloc(80); # for the structure MEM_t
cuda.memset_d32(MEM, 0, 20) # reset the memory buffer
cuda.memcpy_htod(int(MEM), np.double(2000)) # C
cuda.memcpy_htod(int(MEM) + 8, np.double(0)) # global_bias

# 6 1-dimensional arrays
trans_x = np.zeros(nreal).astype(dtype)
trans_x[0], trans_x[1] = 3.14, 19
temp_x1 = np.zeros(nreal).astype(dtype)
temp_x2 = np.zeros(nreal).astype(dtype)
temp_x3 = np.zeros(nreal).astype(dtype)
temp_x4 = np.zeros(nreal).astype(dtype)
temp_x4[0] = 3
norm_x = np.zeros(nreal).astype(dtype)

arrays = [trans_x, temp_x1, temp_x2, temp_x3, temp_x4, norm_x]
for i in xrange(6):
    p = cuda.to_device( arrays[i] )
    GPU_ptrs.append(p)
    cuda.memcpy_htod(int(MEM) + 16 + i * 4, np.intp(p))

# 6 1-dimensional arrays
basic_f = np.zeros(nfunc).astype(dtype)
weight = np.zeros(nfunc).astype(dtype)
sigma = np.zeros(nfunc).astype(dtype)
sigma[0] = 15;
lambd = np.zeros(nfunc).astype(dtype)
bias = np.zeros(nfunc).astype(dtype)
norm_f = np.zeros(nfunc).astype(dtype)
norm_f[2] = 17;

arrays = [basic_f, weight, sigma, lambd, bias, norm_f]
for i in xrange(6):
    p = cuda.to_device( arrays[i] )
    GPU_ptrs.append(p)
    cuda.memcpy_htod(int(MEM) + 16 + 24 + i * 4, np.intp(p))

# 2 2-dimensional arrays (o, g)
for i in xrange(2):
    cuda.memcpy_htod(int(MEM) + 16 + 24 + 24 + i * 4,
            np.intp(int(cuda.to_device(np.zeros(nfunc).astype(np.intp))))) # FIX THIS XXX
# 1 3-dimensional array (l)
cuda.memcpy_htod(int(MEM) + 16 + 24 + 24 + 8,
        np.intp(int(cuda.to_device(np.zeros(nfunc).astype(np.intp))))) # FIX THIS XXX

cuda.memcpy_htod(int(MEM) + 16 + 24 + 24 + 8 + 4, np.int32(42)) # test value

src = ''.join(open('kern1.cu').readlines())
mod = SourceModule(src, arch='sm_13')
cuda.memcpy_htod(mod.get_global('ROB')[0], np.double(1410))
f = mod.get_function('test')
out = np.zeros(10).astype(np.double)
f(MEM, cuda.InOut(out), block=(1,1,1))
print out

sys.exit(0)

f = SourceModule(src, arch='sm_13').get_function('f')

sys.exit(0)

inp = np.linspace(-10,10,512)
out = np.zeros(512)
print inp.astype(np.double)

print f(cuda.In(inp), cuda.Out(out), block=(512,1,1),grid=(1,1), time_kernel=True)

print out



