#!/usr/bin/python
import sys
import pycuda.driver as cuda
import pycuda.gpuarray
import numpy as np
from pycuda.compiler import SourceModule
import pycuda.autoinit

nfunc = 1;
nreal = 50;

dtype = np.double

print 'devices:', cuda.Device.count()


src = ''.join(open('kern1.cu').readlines())
mod = SourceModule(src, arch='sm_13')


#
# INITIALIZATION
#

fpt = ''.join(open('input_data/sphere_func_data.txt', 'r').readlines()).strip().split()
o = np.zeros((nfunc,nreal)).astype(dtype)
for i in xrange(nfunc):
    for j in xrange(nreal):
        o[i,j] = dtype(fpt.pop(0))
print 'o:',o
bias = np.zeros(nfunc).astype(dtype)
bias[0] = -450.0


#
# MEMORY ALLOCATION
#
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
lambd = np.ones(nfunc).astype(dtype)
#bias = np.zeros(nfunc).astype(dtype) # XXX already initialized
norm_f = np.zeros(nfunc).astype(dtype)
norm_f[0] = 17;

arrays = ['basic_f', 'weight', 'sigma', 'bias', 'norm_f']
for var in arrays:
    globals()[var] = cuda.to_device(globals()[var])
    cuda.memcpy_htod(mod.get_global(var)[0], np.intp(globals()[var]))
lambd = cuda.to_device(lambd)
cuda.memcpy_htod(mod.get_global('lambda')[0], np.intp(lambd))

# 2 2-dimensional arrays (o, g)
# MAYBE IT SHOULD BE CASTED to 1-d ARRAY for performance?

#o = np.zeros((nfunc,nreal)).astype(dtype) # XXX already initialized
#o = np.linspace(1,1 +nfunc*nreal-1,nfunc*nreal).reshape((nfunc,nreal))
print o
o_gpu = cuda.to_device(np.zeros(nfunc).astype(np.intp))
o_rows = []
for i in xrange(o.shape[0]):
    row = cuda.to_device(o[i,:])
    o_rows.append(row)
    cuda.memcpy_htod(int(o_gpu) + 4 * i, np.intp(row))
cuda.memcpy_htod(mod.get_global('o')[0], np.intp(o_gpu))

# watch out! 'g' is nreal x nreal
g = np.eye(nreal,nreal).astype(dtype)
#g = np.linspace(21,21 + nreal*nreal-1,nreal*nreal).reshape((nreal,nreal))
print 'g:',g
g_gpu = cuda.to_device(np.zeros(nreal).astype(np.intp))
g_rows = []
for i in xrange(g.shape[0]):
    row = cuda.to_device(g[i,:])
    g_rows.append(row)
    cuda.memcpy_htod(int(g_gpu) + 4 * i, np.intp(row))
cuda.memcpy_htod(mod.get_global('g')[0], np.intp(g_gpu))

# 'l' (3d array) -- casted to 1d
l = np.zeros((nfunc,nreal,nreal)).astype(dtype)
#l = np.linspace(3, 3 + nfunc*nreal*nreal-1,nfunc*nreal*nreal).reshape((nfunc,nreal,nreal))
for i in xrange(nfunc):
    l[i,:,:] = np.eye(nreal)
print 'l:',l
print 'l flatten:', l.flatten()
l_gpu = cuda.to_device(l.flatten().astype(dtype))
print l_gpu
cuda.memcpy_htod(mod.get_global('l_flat')[0], np.intp(l_gpu))


bench = mod.get_function('calc_benchmark_func')

def for_tests(x):
    res = np.zeros(1, dtype)
    bench(cuda.In(x), cuda.Out(res), block=(1,1,1))
    return res[0]


if __name__ == '__main__':
    print '--- Evaluating the benchmark function ---'
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype)
    # x = np.array([-39.3119, 58.8999], dtype) # opt for 2d
    # x = np.array([-39.3119, 58.8999, -46.3224, -74.6515, -16.7997, -80.5441, -10.5935,  24.9694, 89.8384, 9.1119]) # opt for 10D

    x = np.array([-39.3119,58.8999,-46.3224,-74.6515,-16.7997,-80.5441,-10.5935,24.9694,89.8384,9.1119,-10.7443,-27.8558,-12.5806,7.593,74.8127,68.4959,-53.4293,78.8544,-68.5957,63.7432,31.347,-37.5016,33.8929,-88.8045,-78.7719,-66.4944,44.1972,18.3836,26.5212,84.4723,39.1769,-61.4863,-25.6038,-81.1829,58.6958,-30.8386,-72.6725,89.9257,-15.1934,-4.3337,5.343,10.5603,-77.7268,52.0859,40.3944,88.3328,-55.8306,1.3181,36.025,-69.9271]) # opt for 50D

    print 'x:',x
    res = np.zeros(1, dtype)
    bench(cuda.In(x), cuda.Out(res), block=(1,1,1))
    print 'result:',res

    x = np.zeros(50, dtype)
    print 'x:',x
    bench(cuda.In(x), cuda.Out(res), block=(1,1,1))
    print 'result:',res

    sys.exit()


if False:
    f = mod.get_function('test')
    out = np.zeros(10).astype(np.double)
    o_out = np.zeros(nfunc * nreal).astype(dtype)
    g_out = np.zeros(nreal * nreal).astype(dtype)
    l_out = np.zeros(nfunc * nreal * nreal).astype(dtype)
    f(cuda.InOut(out), cuda.Out(o_out), cuda.Out(g_out), cuda.InOut(l_out), block=(1,1,1))
    print 'out:',out
    print o_out
    print g_out
    print l_out



    sys.exit(0)

    f = SourceModule(src, arch='sm_13').get_function('f')

    sys.exit(0)

    inp = np.linspace(-10,10,512)
    out = np.zeros(512)
    print inp.astype(np.double)

    print f(cuda.In(inp), cuda.Out(out), block=(512,1,1),grid=(1,1), time_kernel=True)

    print out



