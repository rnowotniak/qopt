#!/usr/bin/python

import sys
import qopt


def matches(chromo, schema):
    for i in xrange(len(chromo)):
        if schema[i] == '*':
            continue
        if schema[i] != chromo[i]:
            return False
    return True



schema = '******01011****'
#schema = '01001**********'
schema = '01001**********'
schema = '*10011*********'
schema = '010*11*********'
schema = '01*011*********'
schema = '01*011*********'
schema = '010*11*********'
schema = '01*01**********'
#schema = '01001**********'
#schema = '*10011*********'
#schema = '*0********1****'

schema = '01*01'

for i in xrange(2**len(schema)):
    chromo = qopt.int2bin(i, len(schema))
    print i,
    #print chromo,
    print int(matches(chromo, schema))



