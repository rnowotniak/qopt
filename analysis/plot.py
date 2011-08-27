#!/usr/bin/python

import sys
import pylab
import numpy
import getopt

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
    return numpy.matrix(';\n'.join(datalines))

def plot(filename, *args,**kwargs):
    dcol = 0
    vcols = (0,1)
    if args:
        vcols = args

    data = readDataFromFile(filename)

    for vcol in vcols[1:]:
        pylab.plot(data[:,vcols[0]],data[:,vcol],'x-')
    pylab.xlabel(kwargs.get('xlabel'))
    pylab.ylabel(kwargs.get('ylabel'))
    pylab.title(kwargs.get('title'))
    pylab.xlim(xmin=kwargs.get('xmin'))
    pylab.xlim(xmax=kwargs.get('xmax'))
    pylab.ylim(ymin=kwargs.get('ymin'))
    pylab.ylim(ymax=kwargs.get('ymax'))
    return pylab

if __name__ == '__main__':
    #f = sys.argv[1]
    opts,args = getopt.getopt(sys.argv[1:], '', ['--cmpbest'])
    if args:
        f = args[0]
    else:
        f = '/tmp/bla.txt'
    p = plot(f,0,1,2,3,4,5,6,xlabel='generation', title='Plot')
    #p.legend(('evals', 'time'))
    p.savefig('/tmp/zzzz.png')


