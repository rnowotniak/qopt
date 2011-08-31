#!/usr/bin/python

import sys
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

cuda.init()

dev = cuda.Device(0)
print dev.name()
ctx = dev.make_context()

src = '''
    __shared__ double buf[40 * 50];
    __global__ void rosenbrock (double *x, double *out, int dim)
    {
        // memcpy(buf + dim * threadIdx.x, x + dim * threadIdx.x, dim * sizeof(double));
        int i,r;
        double *xx = x; // x + dim * threadIdx.x; // XXX
        double res;
        for (r = 0; r < 1e4; r++) {
            res = 0.0;
            for (i=0; i<dim-1; i++)
            {
                res += 100.0*pow((xx[i]*xx[i]-xx[i+1]),2.0) + 1.0*pow((xx[i]-1.0),2.0);
            }
        }
        out[threadIdx.x] = res;
    }

    __global__ void cec2005complex(double *in, double *out, int dim) {
        int i;
        double x;
        double y;
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

mod = SourceModule(src, arch='sm_13', options=['--use_fast_math', '--ptxas-options=-v'])

fun = mod.get_function('rosenbrock')

data = np.matrix(';'.join(open('/tmp/data.txt').readlines()), np.double)
res = np.zeros((1,40), np.double)
print data
print res

time = fun(cuda.In(data), cuda.Out(res), np.int32(50), block=(1, 1, 1), grid = (1,1), time_kernel = True)
time
print '\n'.join([str(x) for x in res.tolist()[0]])
print '--'
print '%f seconds' % time
print 'performance: %f' % (1e4 * 1 * 1 / time)

cuda.Context.pop()

