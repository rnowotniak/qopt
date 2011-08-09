#!/usr/bin/python

import re,sys
import pylab

logfile = '/tmp/log.txt'
if len(sys.argv) > 1:
    logfile = sys.argv[1]

f = open(logfile, 'r')

while True:
    line = f.readline()
    if not line:
        break
    if line.startswith('## COMMAND:'):
        line = f.readline().strip()
        xmin = float(re.sub(r'.*--xmin (\S+).*', r'\1', line))
        xmax = float(re.sub(r'.*--xmax (\S+).*', r'\1', line))
        expr = re.sub(r'.*--expr (\S+)', r'\1', line)
        break

X = pylab.arange(xmin, xmax, (xmax-xmin)/50)
Y = 1. * X
for i in xrange(X.size):
    x = X[i]
    Y[i] = eval(expr)
pylab.hold(True)
state2 = False
iter = 0
while True:
    line = f.readline()
    if not line:
        break
    if re.match(r'.*# STAT ', line):
        iter += 1
        print 'generation %d' % iter
        pylab.clf()
        pylab.title('generation %d' % iter)
        pylab.plot(X,Y)
        state2 = True
        continue
    if state2:
        if re.match(r'.*\(', line):
            data = re.search(r'\((.*),(.*),(.*)\)', line)
            x,y=[float(x) for x in data.groups()[1:]]
            pylab.plot([x],[y],'o',color='red')
        else:
            state2 = False
            pylab.xlim(xmin,xmax)
            pylab.savefig('/tmp/aa-%03d.png' % iter)



