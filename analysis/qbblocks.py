#!/usr/bin/python

import sys
import re
import numpy as np
import glob
import os.path

bblock     = '01001***************'

def dec2bin(dec, length = None):
    """convert decimal value to binary string"""
    result = ''
    if dec < 0:
        raise ValueError, "Must be a positive integer"
    if dec == 0:
        result = '0'
        if length != None:
            result = result.rjust(length, '0')
        return result
    while dec > 0:
        result = str(dec % 2) + result
        dec = dec >> 1
    if length != None:
        result = result.rjust(length, '0')
    return result


def M(qchromo,schema):
    result = 1.
    for i in xrange(len(schema)):
        if schema[i] == '0':
            result *= np.square(qchromo[0,i])
        elif schema[i] == '1':
            result *= np.square(qchromo[1,i])
    return result

def E(Q,schema):
    sum1 = 0.
    for w in xrange(len(Q)+1):
        sum2 = 0.
        for c in xrange(2**len(Q)):
            bstr = dec2bin(c, len(Q))
            if bstr.count('1') == w:
                #print bstr
                elem = 1.
                for j in xrange(len(bstr)):
                    if bstr[j] == '0':
                        elem *= 1 - M(Q[j], schema)
                    else:
                        elem *= M(Q[j], schema)
                sum2 += elem
        sum1 += 1. * w * sum2
        #print '-'
    return sum1

def V(Q,schema):
    sum1 = 0.
    for w in xrange(len(Q)+1):
        sum2 = 0.
        for c in xrange(2**len(Q)):
            bstr = dec2bin(c, len(Q))
            if bstr.count('1') == w:
                #print bstr
                elem = 1.
                for j in xrange(len(bstr)):
                    if bstr[j] == '0':
                        elem *= 1 - M(Q[j], schema)
                    else:
                        elem *= M(Q[j], schema)
                sum2 += elem
        sum1 += 1. * w*w * sum2
        #print '-'
    return sum1 - E(Q,schema)**2

# q = np.ones((2,20)) * np.sqrt(2)/2
# q[1,1] = 1
# q[0,1] = 0
# q[0,17] = 1
# q[1,17] = 0
# print M(q, bblock)


files = glob.glob('report-qiga.QIGA/log-*')
for fname in files:

    f = open(fname, 'r')
    out = open('/tmp/bb-%s'%os.path.basename(fname),'w')

    iter = 0
    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        match = re.search(r'# iter (\d+)', line)
        if match:
            if iter > 1:
                e = E(Q,bblock)
                v = V(Q,bblock)
                print '%d %f %f %f' % (iter-1, e, v,np.sqrt(v))
                out.write('%d %f %f %f\n' % (iter-1, e, v,np.sqrt(v)))
                out.flush()
            iter = int(match.group(1))
            #print 'iter %d' % (iter-1)
            Q = []
            continue
        match = re.search(r'87# \(\[\[(.*)\]', line)
        if match:
            row1 = match.group(1)
            line2 = f.readline().strip()
            row2 = re.search(r'87#  \[(.*?)\]', line2).group(1)
            m = np.matrix('%s;%s'%(row1,row2))
            Q.append(m)
            #print M(m,bblock)

    out.close()


