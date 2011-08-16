#!/usr/bin/python
import sys
import pycuda.driver as cuda
import pycuda.gpuarray
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit

nreal = 4;
nfunc = 2;

dtype = np.double

print 'devices:', cuda.Device.count()

#   # 1 3-dimensional array (l)
#   cuda.memcpy_htod(int(MEM) + 16 + 24 + 24 + 8,
#           np.intp(int(cuda.to_device(np.zeros(nfunc).astype(np.intp))))) # FIX THIS XXX

src = ''.join(open('kern1.cu').readlines())
mod = SourceModule(src, arch='sm_13')

# memory allocation
cuda.memcpy_htod(mod.get_global('nreal')[0], np.int32(nreal))
cuda.memcpy_htod(mod.get_global('nfunc')[0], np.int32(nfunc))

cuda.memcpy_htod(mod.get_global('C')[0], np.double(2000))
cuda.memcpy_htod(mod.get_global('global_bias')[0], np.double(0))

# 6 1-dimensional arrays
trans_x = np.zeros(nreal).astype(dtype)
trans_x[0], trans_x[1] = 3.14, 19
temp_x1 = np.zeros(nreal).astype(dtype)
temp_x2 = np.zeros(nreal).astype(dtype)
temp_x3 = np.zeros(nreal).astype(dtype)
temp_x4 = np.zeros(nreal).astype(dtype)
temp_x4[1] = 3
norm_x = np.zeros(nreal).astype(dtype)
norm_x[0] = -7;
arrays = ['trans_x', 'temp_x1', 'temp_x2', 'temp_x3', 'temp_x4', 'norm_x']
for var in arrays:
    globals()[var] = cuda.to_device(globals()[var])
    cuda.memcpy_htod(mod.get_global(var)[0], np.intp(globals()[var]))

# 6 1-dimensional arrays
basic_f = np.zeros(nfunc).astype(dtype)
weight = np.zeros(nfunc).astype(dtype)
sigma = np.zeros(nfunc).astype(dtype)
sigma[0] = 15;
lambd = np.zeros(nfunc).astype(dtype)
bias = np.zeros(nfunc).astype(dtype)
norm_f = np.zeros(nfunc).astype(dtype)
norm_f[1] = 17;

arrays = ['basic_f', 'weight', 'sigma', 'bias', 'norm_f']
for var in arrays:
    globals()[var] = cuda.to_device(globals()[var])
    cuda.memcpy_htod(mod.get_global(var)[0], np.intp(globals()[var]))
lambd = cuda.to_device(lambd)
cuda.memcpy_htod(mod.get_global('lambda')[0], np.intp(lambd))

# 2 2-dimensional arrays (o, g)
# MAYBE IT SHOULD BE CASTED to 1-d ARRAY for performance?
o = np.zeros((nfunc,nreal)).astype(dtype)
o = np.linspace(1,1 +nfunc*nreal-1,nfunc*nreal).reshape((2,4))
print o
o_gpu = cuda.to_device(np.zeros(nfunc).astype(np.intp))
o_rows = []
for i in xrange(o.shape[0]):
    row = cuda.to_device(o[i,:])
    o_rows.append(row)
    cuda.memcpy_htod(int(o_gpu) + 4 * i, np.intp(row))
cuda.memcpy_htod(mod.get_global('o')[0], np.intp(o_gpu))

# watch out! 'g' is nreal x nreal
g = np.zeros((nreal,nreal)).astype(dtype)
g = np.linspace(21,21 + nreal*nreal-1,nreal*nreal).reshape((nreal,nreal))
print g
g_gpu = cuda.to_device(np.zeros(nreal).astype(np.intp))
g_rows = []
for i in xrange(g.shape[0]):
    row = cuda.to_device(g[i,:])
    g_rows.append(row)
    cuda.memcpy_htod(int(g_gpu) + 4 * i, np.intp(row))
cuda.memcpy_htod(mod.get_global('g')[0], np.intp(g_gpu))

f = mod.get_function('test')
out = np.zeros(10).astype(np.double)
o_out = np.zeros(nfunc * nreal).astype(dtype)
g_out = np.zeros(nreal * nreal).astype(dtype)
f(cuda.InOut(out), cuda.Out(o_out), cuda.Out(g_out), block=(1,1,1))
print out
print o_out
print g_out

sys.exit(0)

f = SourceModule(src, arch='sm_13').get_function('f')

sys.exit(0)

inp = np.linspace(-10,10,512)
out = np.zeros(512)
print inp.astype(np.double)

print f(cuda.In(inp), cuda.Out(out), block=(512,1,1),grid=(1,1), time_kernel=True)

print out



