#!/usr/bin/python

import sys
import re
import numpy

numpy.set_printoptions(linewidth=float('inf'))

alldata = []
for fname in sys.argv[1:]:
    f = open(fname, 'r')
    m = []
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        match = re.search(r'.*# STAT(\s+\S+)(\s+\S+)(\s+\S+)(\s+\S+)(\s+\S+)(\s+\S+)(\s+\S+)', line)
        if match:
            m.append([float(x) for x in match.groups()])
    f.close()
    m = numpy.matrix(m)
    alldata.append(m)

res=alldata[0]
for i in xrange(1,len(alldata)):
    res += alldata[i]
res /= len(alldata)

for row in res:
    print '%d %d  %.03f   %g  %g  %g  %g' % tuple(*row.tolist())

