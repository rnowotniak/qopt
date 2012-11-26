#!/usr/bin/python

#
#
# Wczytuje dane z plikow /var/tmp/*-f*/run*.txt
#

import sys
import re
import numpy as np
import glob
import qopt.problems.CEC2005.cec2005 as cec2005

p = re.compile(r"""^\d+ [^ ]+$""")


def readFile(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    lines = filter(lambda line: p.match(line), lines)
    arr = np.matrix( ';'.join(lines) )
    return arr

def success_performance(method, fnum):
    files = glob.glob('/var/tmp/%s-f%s/run*.txt' % (method, fnum))
    if len(files) == 0:
        return -1
    ok = 0
    fevalssum = 0
    for f in files:
        m = readFile(f)
        fun = cec2005.FUNCTIONS['f%s' % fnum]
        isok = m[-1,1] <= fun['opt'] + fun['accuracy']
        if isok:
            ok += 1
            fevals = filter(lambda row: row[0][0,1] <= fun['opt'] + fun['accuracy'], m)[0][0,0]
            # print f, fevals
            fevalssum += fevals
    if ok == 0:
        return -1
    fevalssum /= ok
    fevalssum *= len(files) / ok
    return fevalssum

def success_rate(method, fnum):
    files = glob.glob('/var/tmp/%s-f%s/run*.txt' % (method, fnum))
    if len(files) == 0:
        return -1
    ok = 0
    for f in files:
        m = readFile(f)
        fun = cec2005.FUNCTIONS['f%s' % fnum]
        ok += m[-1,1] <= fun['opt'] + fun['accuracy']
    return 1. * ok / len(files)

for fnum in xrange(1,26):
    print '%d %.f%% %d' % (fnum, success_rate('de', fnum) * 100, success_performance('de', fnum))


#
# Ponizej mamy generowanie wykresow porownujacych poszczegolne metody dla funkcji 1-25
#
sys.exit(0)


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

