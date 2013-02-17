#!/usr/bin/python

import sys
import matplotlib.pyplot as plt
import numpy as np

def readDataFromFile(filename):
    f = open(filename,'r')
    datalines = []
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            continue
        datalines.append(line)
    return np.matrix(';\n'.join(datalines))

sgadata = readDataFromFile('bb-sga-avg.txt')
qigadata = readDataFromFile('bb-qiga-avg.txt')
qigaACTUALdata = readDataFromFile('actual.txt')

plt.plot(qigadata[:,0],qigadata[:,1], 'ro-', label='Expected Propagation in QIGA')
plt.plot(qigadata[:,0],qigadata[:,1]+qigadata[:,2], 'r-')
plt.plot(qigadata[:,0],qigadata[:,1]-qigadata[:,2], 'r-')

plt.fill_between(
        qigadata[:,0].transpose().tolist()[0],
        (qigadata[:,1]-qigadata[:,2]).transpose().tolist()[0],
        (qigadata[:,1]+qigadata[:,2]).transpose().tolist()[0],
        color="r", alpha=0.2)

plt.errorbar(
        qigadata[:,0].transpose().tolist()[0],
        qigadata[:,1].transpose().tolist()[0],
        qigadata[:,2].transpose().tolist()[0],
        fmt='ro', label='')

plt.plot(qigaACTUALdata[:,0],qigaACTUALdata[:,1], 'bs-', label='Actual Propagation in QIGA')

plt.plot(sgadata[:,0],sgadata[:,1], 'gx-', label='Actual Propagation in SGA')
plt.title('Building Block Propagation Comparison')
plt.xlabel('generation number')
plt.ylabel('number of chromosomes matching the schema')
plt.grid(True)
plt.legend(loc='lower right')

plt.text(110, 5, "Subject to Holland's\nschema theorem",ha='center')
plt.annotate("", (130,4.2), xytext=(110,4.8),
        arrowprops={'facecolor':'black',})#, ha='center')

plt.savefig('/tmp/bbcmpplot.pdf', bbox_inches='tight')

