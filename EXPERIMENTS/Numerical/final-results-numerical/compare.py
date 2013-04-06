#!/usr/bin/python

import sys
import numpy as np
import qopt.problems
import operator

data = np.matrix(np.load('pso-cmaes-ga-nelder-dim2.npy'))
numfuncs = data.shape[0]
numalgs = data.shape[1] / 4

HIGHLIGHT = "\033[1;31m"
NORMAL = "\033[0m"

print 'fnum opt    PSO        CMAES       GA     NelderMead'
print '----------------------------------------------------'



algs = ['PSO', 'CMAES', 'GA', 'NelderMead']

ranking = {}
for alg in algs:
    ranking[alg] = 0

for fnum in xrange(1, numfuncs + 1):
    fobj = qopt.problems.CEC2013(fnum)
    optval = fobj.evaluate(fobj.optimum)
    print '%2d %5g' % (fnum, optval),
    row = data[fnum - 1]
    best = min( row[0,0], row[0,4], row[0,8], row[0,12] )
    for field in xrange(numalgs * 4):
        if field % 4 != 0:
            continue
        sys.stdout.write(NORMAL)

        val = row.tolist()[0][field]
        if field % 4 == 0:
            if val == best:
                ranking[algs[field / 4]] += 1
                sys.stdout.write(HIGHLIGHT)
        print '%10.4g ' % val,
        sys.stdout.write(NORMAL)
    print

print 
print 'Ranking:'
print '--------'

table = sorted(ranking.iteritems(), key = operator.itemgetter(1), reverse=True)
for row in table:
    print row[0], row[1]

