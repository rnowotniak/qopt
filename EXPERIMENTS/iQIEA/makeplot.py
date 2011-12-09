#!/usr/bin/python

import sys

import matplotlib
matplotlib.use('cairo')

from pylab import *
import sys

#from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import sys

print
print 'Generating the plot...'
sys.stdout.flush()

os.system('bash -c "python %s/avg.py /tmp/qopt/iQIEA.py/output.* > /tmp/qopt/iQIEA.py/avg.log"' % \
        (os.path.dirname(os.path.realpath(__file__))))

fname = '/tmp/qopt/iQIEA.py'

# m1 = np.matrix(';'.join(open(fname+'-orig/avg.log','r').readlines()))
m2 = np.matrix(';'.join(open(fname+'/avg.log','r').readlines()))

# plot(m1[:,0], m1[:,1], 'o-', label='Original RCQiEA algorithm', color='black', markersize=8, markerfacecolor='white', markevery=30, linewidth=3)
plot(m2[:,0], m2[:,1], 'rs-', label=r'iQiEA($\xi,\delta$)', markersize=8, markevery=30, linewidth=3)
legend(loc='upper right')
#title(r'Comparison for function $f_%s$'%fname)
xlabel('Objective function evaluation count')
ylabel('Objective function value')
# if fname == '1':
#     xlim([0,2000])
#     ylim([0,10000])
# elif fname == '2':
#     xlim([0,2500])
#     ylim([0,1000])
# elif fname == '3':
#     xlim([0,8000])
#     ylim([0,500])
# elif fname == '4':
xlim([0,4000])
ylim([0,16])
grid(True)

# if True: #fname != '3':
#     savefig('/tmp/cmp-f%s.pdf'%fname, bbox_inches='tight')#, dpi=600)
#     sys.exit(0)

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
# subfig = imread('cmp-f3sub.png')
# subfig = OffsetImage(subfig, zoom=0.45)
# ab = AnnotationBbox(subfig, (250, 350),
#        xybox=(5500,275),
#        xycoords='data',
#        boxcoords="data",
#        #pad=0.5,
#        arrowprops=dict(arrowstyle="->", lw=3),
#            #connectionstyle="angle,angleA=0,angleB=90,rad=3")
#        )
# 
# gca().add_artist(ab)

#fig = plt.gcf()
#fig.clf()
#ax = plt.subplot(111)
#ax.add_artist(ab)

#imshow(subfig, aspect='auto', origin='upper', extent=(1000, 7000, 100,400))

#show()
savefig('/tmp/qopt/iQIEA.py/plot.png', bbox_inches='tight')

print 
print '   The plot has been generated'
print 

