#!/usr/bin/python

#
#
# Wczytuje dane z plikow /var/tmp/*-f*/run*.txt
#

import sys
import re
import numpy as np
import glob

p = re.compile(r"""^\d+ [^ ]+$""")

def readFile(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    lines = filter(lambda line: p.match(line), lines)
    arr = np.matrix( ';'.join(lines) )
    return arr

import matplotlib.pyplot as plt

plt.hold(True)

def method(fnum, name):
    mces = []
    for f in glob.glob('/var/tmp/%s-f%s/run*.txt' % (name, str(fnum))):
        m = readFile(f)
        if name == 'sga':
            m[:,0] *= 100.
        mces.append(m)
    if len(mces) == 0:
        return
    avg = np.array(mces).sum(axis = 0) / len(mces)
    plt.grid(True)
    plt.plot(avg[:,0], avg[:,1], 'x-', label=name, markersize = 3, markevery = 50)


def fun(fnum):
    plt.cla()
    method(fnum, 'pso')
    method(fnum, 'sga')
    method(fnum, 'bat')
    method(fnum, 'de')
    plt.legend(loc=1)
    plt.savefig('/tmp/f%s-cmp.png' % fnum, bbox_inches = 'tight')

for n in xrange(1,26):
    print n
    fun(n)

