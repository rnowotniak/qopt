#!/usr/bin/python

import sys

from mpl_toolkits.mplot3d import Axes3D
from numpy import *
import matplotlib.pyplot as plt
import pylab

import numpy as np


cpu=matrix(';'.join(open('cpu-time').readlines()))
gpu=matrix(';'.join(open('gpu-time2.txt').readlines()))

gpu[:,1] *= 2

#gpu=matrix(';'.join(open('/tmp/gpu').readlines()))

#pylab.xticks((1,2),('CPU, 100 experiments\n(sequential implementation)','GPU, 1280 experiments\n(parallel implementation)'))
#pylab.ylabel('distribution of best fitness value in independent experiments\n(500 generations of 10 quantum individuals each)')
pylab.ylabel('Time (s)')
pylab.xlabel('Number of Populations')
pylab.title('Performance Comparison')

pylab.grid(True)

#pylab.plot(gpu[:,0], gpu[:,1], 'ro-', label='GPU (nVidia GTX 295)')
#pylab.plot(cpu[:,0], cpu[:,1], 's-', label='CPU (Intel Core i7)')

pylab.plot(cpu[:,0], cpu[:,1], 'x-', label='CPU (Intel Core i7)')
pylab.plot(gpu[:,0], gpu[:,1], 'rs-', label='GPU (nVidia GTX 295)')
pylab.legend(loc=1)
pylab.xlim((50,10000))
pylab.savefig('/tmp/time-cmp.pdf', bbox_inches='tight')

pylab.cla()
pylab.loglog(cpu[:,0], cpu[:,1], 'x-', label='CPU (Intel Core i7)')
pylab.loglog(gpu[:,0], gpu[:,1], 'rs-', label='GPU (nVidia GTX 295)')
pylab.legend(loc=1)
pylab.minorticks_on()
pylab.xlim((50,10000))
pylab.grid(True, linestyle='-')
pylab.grid(True, which='minor', linestyle='-.')
pylab.ylabel('Time (s)')
pylab.xlabel('Number of Populations')
pylab.title('Performance Comparison')
pylab.savefig('/tmp/time-cmp-log.pdf', bbox_inches='tight')


