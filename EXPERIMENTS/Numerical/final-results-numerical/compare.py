#!/usr/bin/python

import sys
import numpy as np
import qopt.problems
import operator

myrqiea2 = np.matrix(np.load('myrqiea2-dim2.npy'))
data = np.matrix(np.load('/tmp/multiprocessing-pso-cmaes-ga-nelder-dim2.npy'))

algs = ['MyRQIEA2', 'PSO', 'CMAES', 'GA', 'NelderMead']
#algs = [ 'PSO', 'CMAES', 'GA', 'NelderMead']

MEAN_ONLY = True

data = np.hstack((myrqiea2, data))

numfuncs = data.shape[0]
numalgs = data.shape[1] / 4 # / number of fields

HIGHLIGHT = "\033[1;31m"
NORMAL = "\033[0m"
#HIGHLIGHT = ""
#NORMAL = ""


ranking = {}
for alg in algs:
    ranking[alg] = 0.



print 'fnum opt   ' + ''.join(['    %s    ' % alg for alg in algs])
print '--------------------------------------------------------------------------'

for fnum in xrange(1, numfuncs + 1):
    fobj = qopt.problems.CEC2013(fnum)
    optval = fobj.evaluate(fobj.optimum)
    print '%2d %5g ' % (fnum, optval),
    row = data[fnum - 1]
    best = min( row[0,0], row[0,4], row[0,8], row[0,12], row[0,16] )
    rowranking = []
    for field in xrange(numalgs * 4):
        if MEAN_ONLY and field % 4 != 0:
            continue
        sys.stdout.write(NORMAL)

        val = row.tolist()[0][field]
        if field % 4 == 0: # this field is mean value
            # make ranking for this row
            rowranking.append((algs[field/4], val))

            if val == best:
                #ranking[algs[field / 4]] += 1
                sys.stdout.write(HIGHLIGHT)
        print '%10g ' % val,
        sys.stdout.write(NORMAL)
    print
    rowranking = sorted(rowranking, cmp = lambda a,b: cmp(a[1],b[1]))
    rowranking = map(lambda i: i[0], rowranking)
    #print rowranking
    for alg in algs:
        pass
        ranking[alg] += rowranking.index(alg) + 1

for alg in ranking:
    ranking[alg] /= 28

print ranking

print 
print 'Ranking:'
print '--------'

table = sorted(ranking.iteritems(), key = operator.itemgetter(1), reverse = False)
for row in table:
    print row[0], row[1]

