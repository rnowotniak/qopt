#!/usr/bin/python

import sys
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

cuda.init()

dev = cuda.Device(1)
print dev.name()
ctx = dev.make_context()

src = '''
    // __shared__ float buf[40 * 50];
    // Using shared memory gave worst results unfortunatelly

    __global__ void rosenbrock (float *x, float *out, int dim)
    {
        int i, r;
        float *xx = x + dim * threadIdx.x;

        float res;
        for (r = 0; r < 1e3; r++) {
            res = 0.0;
            for (i=0; i<dim-1; i++)
            {
                res += 100.0f*pow((xx[i]*xx[i]-xx[i+1]),2.0f) + 1.0f*pow((xx[i]-1.0f),2.0f);
            }
        }
        out[threadIdx.x] = res;
    }

    __global__ void cec2005complex(double *in, double *out, int dim) {
        int i;
        float x;
        float y;
        for (i = 0; i < 1e6; i++) {
            x = 5.55;
            x = x + x;
            x = x / 2;
            x = x * x;
            x = sqrt(x);
            x = log(x);
            x = exp(x);
            y = x/x;
        }
        out[threadIdx.x] = y;
        // return y;
    }

'''

mod = SourceModule(src, arch='sm_13', options=['--ptxas-options=-v'])

fun = mod.get_function('rosenbrock')

data = np.matrix(';'.join(open('simple-rosenbrock-pycuda-speedup-test-data.txt').readlines()), np.float32)
print data.shape
res = np.zeros((1,256), np.float32)
print data
print res

time = fun(cuda.In(data), cuda.Out(res), np.int32(50), block=(256, 1, 1), grid = (1,300), time_kernel = True)
time *= 1e6
evals = 1e3 * 256 * 300
print '\n'.join([str(x) for x in res.tolist()[0]])
print '--'
print '%d evals' % evals
print '%f microseconds' % time
print 'performance: %f (evals / microsecond)' % (evals / time)

cuda.Context.pop()

