#!/usr/bin/python

import sys
import math

N = 2

def f1(x):
    result = 0.
    for j in xrange(len(x)):
        result += x[j]**2
    return result

def f2(x):
    result = 0.
    for j in xrange(len(x)):
        result += abs(x[j])
    prod = 1.
    for j in xrange(len(x)):
        prod *= abs(x[j])
    result += prod
    return result

#print f2([0,0])
#print f2([0,5])
#print f2([3,0])
#print f2([9.5,9.5])
#sys.exit(0)

def f3(x):
    result = 0.
    for j in xrange(len(x)):
        result += x[j]**2
    result *= 1./4000
    p = 1.
    for j in xrange(len(x)):
        p *= math.cos(x[j]/math.sqrt(j+1))
    result = result - p + 1
    return result

#print f3([0,0])
#print f3([0,10])
#print f3([10,0])
#print f3([9.5,9.5])
#sys.exit(0)

def f4(x):
    result = 0
    for j in xrange(len(x)):
        result += x[j]**2
    result *= 1./len(x)
    result = -20 * math.exp(-0.2 * math.sqrt(result))
    b = 0
    for j in xrange(len(x)):
        b += math.cos(2.*math.pi*x[j])
    b *= 1./len(x)
    result -= math.exp(b)
    result += 20 + math.e
    return result

# x = [0] * N
# x[1] = 0
# print f4(x)
# x[1] = 1
# print f4(x)
# x[1] = 0.3
# print f4(x)
# sys.exit(0)

if __name__ == '__main__':
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    xi=np.linspace(-30,30,50)
    yi=np.linspace(-30,30,50)
    X,Y=np.meshgrid(xi,yi)
    Z=X*X+Y*Y
    ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap = cm.jet, shade=False)
    plt.savefig('/tmp/f1.pdf', bbox_inches='tight')


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlabel(r'$x_1$')
#ax.set_ylabel(r'$x_2$')
#xi=np.linspace(-10,10,50)
#yi=np.linspace(-10,10,50)
#X,Y=np.meshgrid(xi,yi)
#Z=abs(X)+abs(Y) + abs(X)*abs(Y)
#ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap = cm.jet, shade=False)
#plt.savefig('/tmp/f2.pdf', bbox_inches='tight')
#

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlabel(r'$x_1$')
#ax.set_ylabel(r'$x_2$')
#xi=np.linspace(-8,8,80)
#yi=np.linspace(-8,8,80)
#X,Y=np.meshgrid(xi,yi)
#Z= 1./4000 * (X*X + Y*Y) - np.cos(X) * np.cos(Y/math.sqrt(2)) + 1
#ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap = cm.jet, shade=False)
#plt.savefig('/tmp/f3.pdf', bbox_inches='tight')


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlabel(r'$x_1$')
#ax.set_ylabel(r'$x_2$')
#xi=np.linspace(-6,6,50)
#yi=np.linspace(-6,6,50)
#X,Y=np.meshgrid(xi,yi)
#Z= -20. * np.exp(-0.2*np.sqrt(1./2*(X*X + Y*Y))) - np.exp( 1./2 * (np.cos(2.*np.pi*X)+np.cos(2.*np.pi*Y))  ) + 20 +np.e
#ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap = cm.jet, shade=False)
#plt.savefig('/tmp/f4.pdf', bbox_inches='tight')
#





