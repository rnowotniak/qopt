#!/usr/bin/python
# -*- coding: utf-8 -*-

'''
compressor
==========

Compresses data (e.g. averaging)
'''

import os
import math


def avg_stats(xCol, yCol, numFiles, filePrefix='', fileSuffix='', filePath='.', delim=',', outName=None):
    '''
    This function can take several data files and transfer them into a file that is formatted
    ready to be plotted by gnuplot (each line is "<x> <mean of y vals> <std of y vals>").
    Data files should be named using an index starting by 1 and all have the same prefix and/or suffix arround that index.

    In gnuplot, you could then say "plot outName smooth unique with yerrorlines"

    :param int xCol: x column
    :param int yCol: y column
    :param int numFiles: the number of files to average over
    :param string filePrefix: prefix in filenames
    :param string fileSuffix: suffix in filenames
    :param string filePath: path to files
    :param string delim: delimiter used between columns
    :param string outName: Name of the result file, defaults to <filePrefix><yCol><fileSuffix>.out
    '''

    assert os.path.exists(filePath), 'File path %s does not exist' % filePath
    if len(filePath) > 1 and not filePath.endswith("/"): filePath += "/"
    if numFiles is None:
        numFiles = len([f for f in os.listdir(filePath) if f.endswith(fileSuffix) and f.startswith(filePrefix)])
    assert numFiles, 'numFiles is zero or not set'
    assert xCol, 'xCol is not set'
    assert yCol, 'yCol is not set'

    # ---- First,  we'll collect all y values from all files
    d = {} # this will store a list of y values for each x
    for i in xrange(1, numFiles+1, 1):
        f = open(filePath + filePrefix + str(i) + fileSuffix, 'r')
        hasMoreRows = True
        while hasMoreRows:
            s = f.readline().strip().split(delim)
            if s == ['']:
                hasMoreRows = False
            else:
                # disregard comments and unsuitable lines
                if s[0].startswith('#') or len(s) < xCol or len(s) < yCol:
                    continue
                x = s[xCol-1].strip()
                if not x == '' and not d.has_key(x):
                    d[x] = []
                try:
                    # we assume that y values are numeric! Also,other
                    # errors might happen here when file is corrupted
                    d[x].append(float(s[int(yCol)-1]))
                except Exception, e:
                    print "ERROR"
        f.close()

    # ---- Then, we compute mean and standard deviation for the v values for
    #      each x and write them to target file
    if outName is None: outName = '%s%s%s%s.out' % (filePath, filePrefix, str(yCol), fileSuffix)
    out = open(outName, 'w')
    keys = d.keys()
    keys.sort()
    for x in keys:
        # -- mean
        sum = 0.0
        for y in d[x]:
            sum += y
        mean = sum / float(len(d[x]))
        # -- standard deviation (std) or standard error (ste)
        #    On the difference between them, see the very readable intro at
        #    http://ww1.cpa-apc.org:8080/publications/archives/PDF/1996/Oct/strein2.pdf
        std = 0.0
        for y in d[x]:
            std += math.pow(y - mean, 2)
        std /= len(d[x])-1
        std = math.sqrt(std)
        ste = std / math.sqrt(len(d[x])) # we're not using this, #40 should give a configurable choice
        out.write('%s %f %f\n' % (str(x), mean, std))

    out.close()

