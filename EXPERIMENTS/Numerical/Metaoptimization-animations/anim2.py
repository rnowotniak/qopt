#!/usr/bin/python
#encoding: utf-8

import copy
import sys
import re
import glob

f=open('z.txt', 'r')

LIMIT = 50

data={}

while True:
    line = f.readline()
    if not line:
        break
    params = line.strip()
    avg = 0.
    for n in xrange(50):
        v = int(f.readline())
        if v == 50010:
            v = 100010
        if n < LIMIT:
            avg += v
    avg /= LIMIT
    print params, avg
    data[params] = avg
#sys.exit(0)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.gca()
XI = np.arange(0, 1.1, 0.1)
DELTA = np.arange(0.1, 1.1, 0.1)
#print DELTA
#sys.exit(0)
XI, DELTA = np.meshgrid(XI, DELTA)
print XI.shape
print DELTA.shape
Z = copy.deepcopy(XI)
for i in xrange(Z.shape[0]):
    for j in xrange(Z.shape[1]):
        Z[i][j] = data['%s-%s'%(XI[0,:][j], DELTA[:,0][i])]
#print Z
#sys.exit(0)
#print R
#sys.exit(0)
#Z = np.sin(R)


#surf = ax.plot_surface(XI, DELTA, Z, rstride=1, cstride=1, cmap=cm.jet,
#                        linewidth=0, antialiased=False)
surf = ax.contourf(DELTA, XI, Z, 50, cmap=cm.jet_r)
plt.grid(True)
#plt.title(r'Meta-fitness landscape for function $f_3$')
#plt.annotate('x', xy=(0.7,0.6), size='xx-large', color='white')
#plt.plot([0.8], [0.3], '*', color='white', markersize=10)
#ax.set_zlim3d(-1.01, 1.01)
ax.set_ylabel(u'prawdopodobieństwo krzyżowania $\\xi$')
ax.set_xlabel(u'współczynnik kontrakcji $\\delta$')
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

LIMIT = int(sys.argv[1])
lusStepsFile = sys.argv[2]

points = np.matrix(';'.join(open(glob.glob(lusStepsFile)[0], 'r').readlines()[:LIMIT]))
for i in xrange(points.shape[0] - 1):
    ax.annotate('', xy=(points[i+1,1], points[i+1,0]), xytext=(points[i,1], points[i,0]),
            arrowprops={'arrowstyle':'->', 'connectionstyle':'arc3', 'linewidth': 2})
ax.plot(points[:,1], points[:,0], 'bo', markersize = 5, markerfacecolor='white')
#ax.plot(points[:,0], points[:,1], 's-.', linewidth = 3, color = '#00F000')

#ax.w_zaxis.set_major_locator(LinearLocator(10))
#ax.w_zaxis.set_major_formatter(FormatStrFormatter('%g'))

cbar = fig.colorbar(surf, shrink=0.5, aspect=5, format='%d')
#cbar.set_ticks(ticks=np.linspace(20000,100000,9))
#cbar.set_clim(10000, 100000)
#cbar.set_norm(colors.Normalize(clip=False))
#cbar.set_label(r'metafitness $\widetilde{f}(\xi,\delta)$')#, va='top')#, ha='left')
plt.text(1.02, 0.8, r'metaocena $\widetilde{f}(\xi,\delta)$')#, va='top')#, ha='left')

#plt.savefig('/tmp/metafitness-%03d.pdf'%LIMIT, bbox_inches='tight', pad_inches=.2)
plt.savefig('/tmp/anim2.png', dpi=150, bbox_inches='tight', pad_inches=.2)
#plt.show()


