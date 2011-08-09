#!/usr/bin/python

# graph.py [opts] -o /tmp/graph.png logfile.txt

from pylab import *
import sys

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


gen = []
best = []
avg = []
worst = []
f=open('/tmp/log.txt', 'r')
while True:
    line = f.readline().strip()
    if not line:
        break
    if line.startswith('#'):
        continue
    data = line.split()
    print data
    gen.append(int(data[0]))
    best.append(float(data[3]))
    avg.append(float(data[4]))
    worst.append(float(data[5]))

plot(gen, best, 'g+-', linewidth=1.0, label='best')
plot(gen, avg, 'b+-', linewidth=1.0, label='avg')
plot(gen, worst, 'r+-', linewidth=1.0, label='worst')
fill_between(gen, worst, best, color="g", alpha=0.2)

xlabel('generation')
ylabel('fitness')
title('Plot of evolution')
grid(True)
legend(loc='lower center')
#savefig('/tmp/bla.png')
#savefig('/tmp/bla.ps')
savefig('/tmp/bla.pdf')
#show()

