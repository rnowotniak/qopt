#!/usr/bin/python

import sys
import numpy as np
import cec2005


# sequential test
if True:
    print '---[ Sequential test ]---'
    for fnum in xrange(1,7):
        print 'fnum: ', fnum
        if fnum == 4:
            print 'noisy function (skipping)'
            continue
        fpt = open('input_data/test_data_func%d.txt' % fnum).readlines()
        i = 0
        xargs = []
        vals = []
        # read x_args and correct function values
        while True:
            m = np.matrix(fpt[0], np.double)
            if m.size == 1:
                break
            fpt.pop(0)
            xargs.append(m)
        for i in xrange(len(xargs)):
            m = np.double(fpt[0])
            fpt.pop(0)
            vals.append(m)

        f = getattr(cec2005, 'f%d' % fnum)
        for i in xrange(len(xargs)):
            calculated = f(xargs[i])
            print '%e %e' % (calculated[0], vals[i])
            assert np.allclose(calculated, vals[i])
    #sys.exit(0)


# parallel test
if True:
    print '---[ Parallel test ]---'
    for fnum in xrange(1,7):
        print 'fnum: ', fnum
        if fnum == 4:
            print 'noisy function (skipping)'
            continue
        fpt = open('input_data/test_data_func%d.txt' % fnum).readlines()
        i = 0
        xargs = []
        vals = []
        # read x_args and correct function values
        while True:
            m = np.matrix(fpt[0], np.double)
            if m.size == 1:
                break
            fpt.pop(0)
            xargs.append(m)
        for i in xrange(len(xargs)):
            m = np.double(fpt[0])
            fpt.pop(0)
            vals.append(m)

        m = np.matrix(np.zeros((len(xargs),xargs[0].size)))
        for i in xrange(len(xargs)):
            m[i,:] = np.matrix(xargs[i])
        xargs = m

        f = getattr(cec2005, 'f%d' % fnum)
        calculated = f(xargs)

        for i in xrange(xargs.shape[0]):
            print calculated[i], vals[i]
            assert np.allclose(calculated[i], vals[i])


# test execution time
print '---[ Execution time test ]---'
cpuperf = 512000. / 25414375 # On CPU, evaluation of 512000 takes approx. 25 seconds
blocks = 100
blocksize = 100
rep = 10
gputime = cec2005.test_time(blocksize, blocks, rep) * 1e6
gpuperf = 1.0 * blocks * blocksize * rep / gputime
print 'GPU time: %f microseconds' % gputime
print 'cpuperf: %f' % cpuperf
print 'gpuperf: %f' % gpuperf
print 'SPEED-UP: %g' % (gpuperf / cpuperf)
assert gpuperf > cpuperf

print 'Well done.'

import pycuda.driver as cuda
cuda.Context.pop()

