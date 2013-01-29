#!/usr/bin/python

import sys


import numpy
import numpy.random
import matplotlib.pyplot as plt
nbPoints=500

# data
points = 900
mean = [-2.,-3.]
theta = 20. * numpy.pi / 180
d1 = .1
d2 = 1


# code

rot = numpy.matrix([
[numpy.cos(theta), -numpy.sin(theta)],
[numpy.sin(theta), numpy.cos(theta)]])
B = numpy.matrix(
        [[1,0],
        [0,1]])
B = B * rot
D = numpy.matrix(
        [[d1,0],
        [0,d2]])
cov = B*D*D*B.T
xy = numpy.random.multivariate_normal(mean,cov,points).T

fig=plt.figure()
axis=fig.add_subplot(111,aspect='equal')
plt.xlim((-10,10))
plt.ylim((-10,10))
plt.scatter(xy[0,:],xy[1,:],c='#FF4500', alpha = .2)

import random,math

# METHOD 2
def box_muller(how_many):
    res = []
    for i in xrange(how_many):
        u1 = random.random()
        u2 = random.random()
        z1 = math.sqrt(-2.*math.log(u1)) * math.cos(2.*math.pi*u2)
        res.append(z1)
    return numpy.matrix(res)

u =  d1 * box_muller(points)
v =  d2 * box_muller(points)

m = numpy.vstack((u,v))
m = rot * m

m[0,:] += mean[0]
m[1,:] += mean[1]

plt.scatter(m.tolist()[0],m.tolist()[1],c='green', alpha = .2)

plt.savefig('/tmp/blaa.pdf')


