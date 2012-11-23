#!/usr/bin/python

import sys
import re
import numpy as np
import glob
import os.path


chromlen = 15

def dec2binary(dec, length = None):
    res = ''
    while dec > 0:
        res = str(dec % 2) + res
        dec = dec >> 1
    if length != None:
        res = res.rjust(length, '0')
    return res

def dec2ternary(dec, length = None):
    res = ''
    while dec > 0:
        res = str(dec % 3) + res
        dec = dec // 3
    if length != None:
        res = res.rjust(length, '0')
    return res

def order(schema):
    return len(schema) - schema.count('*')

def deflength(schema):
    s = schema.replace('1', '0')
    i = s.find('0')
    if i == -1:
        return i
    return s.rfind('0') - i

def fitness(schema):
    sum_ = 0.
    c = list(schema[:])
    wildcards_positions = [i for i,x in enumerate(schema) if x == '*']
    #print 'matching chromosomes:'
    for n in xrange(2**len(wildcards_positions)):
        possibility = dec2binary(n, len(wildcards_positions))
        for i in xrange(len(possibility)):
            c[wildcards_positions[i]] = possibility[i]
        chromo = ''.join(c)
        # sum_ += chromo_fitness(chromo)
    #return sum_ / 2**len(wildcards_positions)
    return str(wildcards_positions)


for n in xrange(3**chromlen):
    schema = dec2ternary(n, chromlen).replace('2','*')
    o = order(schema)
    if o > 3 or o < 2:
        continue
    dl = deflength(schema)
    if dl > 2:
        continue
    print schema, o, dl, fitness(schema)
    #print '--'

sys.exit(0)

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

if __name__ == '__main__':

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


