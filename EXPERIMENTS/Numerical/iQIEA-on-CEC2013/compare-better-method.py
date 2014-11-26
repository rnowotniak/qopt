#!/usr/bin/python

import sys
import numpy as np
import qopt.problems
import operator
import glob

DIM = 50

#iqiea = np.matrix(np.load('multiprocessing-iqiea-dim%d.npy' % DIM))
#qiea1 = np.matrix(np.load('multiprocessing-qiea1-dim%d.npy' % DIM))
#qiea2 = np.matrix(np.load('multiprocessing-qiea2-dim%d.npy' % DIM))
#qiea2=qiea1
#myrqiea2 = np.matrix(np.load('myrqiea2-dim%d.npy' % DIM))
data = np.matrix(np.load('multiprocessing-pso-cmaes-ga-nelder-dim%d.npy' % DIM))

algs = ['PSO', 'CMAES', 'GA', 'NelderMead'] # , 'QIEA2']
#algs = [ 'PSO', 'CMAES', 'GA', 'NelderMead']

data = np.hstack((data, ))  # qiea2))

iQIEAs = {}
#for iQIEA_file in [f.strip() for f in open('/tmp/blabla.txt').readlines()]:
#for iQIEA_file in glob.glob('multiprocessing-iqiea-dim%d-*.npy' % DIM):
for iQIEA_file in sys.argv[1:]:
    xi, delta = iQIEA_file.replace('multiprocessing-iqiea-dim%d-' % DIM, '').replace('.npy', '').split('-')
    #print xi, delta
    algs.append( 'iQIEA-%s,%s' % (xi, delta))
    data = np.hstack((data, np.matrix(np.load(iQIEA_file)) ))

#sys.exit(0)

MEAN_ONLY = True


numfuncs = data.shape[0]
numalgs = data.shape[1] / 4 # / number of fields

HIGHLIGHT = "\033[7;31m"
NORMAL = "\033[0m"
#HIGHLIGHT = ""
#NORMAL = ""


ranking1 = {}
ranking2 = {}
for alg in algs:
    ranking1[alg] = 0.
    ranking2[alg] = 0.



print 'fnum opt   ' + ''.join(['    %s    ' % alg for alg in algs])
print '--------------------------------------------------------------------------'

for fnum in xrange(1, numfuncs + 1):
    fobj = qopt.problems.CEC2013(fnum)
    optval = fobj.evaluate(fobj.optimum)
    print '%2d %5g ' % (fnum, optval),
    row = data[fnum - 1]
    best = min([row[0,4*a] for a in xrange(numalgs )])
    #best = min( row[0,0], row[0,4], row[0,8], row[0,12], row[0,16] )
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
                ranking1[algs[field / 4]] += 1
                sys.stdout.write(HIGHLIGHT)
        print '%10g ' % val,
        sys.stdout.write(NORMAL)
    print
    rowranking = sorted(rowranking, cmp = lambda a,b: cmp(a[1],b[1]))
    rowranking = map(lambda i: i[0], rowranking)
    print rowranking
    for alg in algs:
        ranking2[alg] += rowranking.index(alg) + 1

for alg in ranking2:
    pass
    ranking2[alg] /= 28

print ranking1

print 
print 'Ranking1:'
print '--------'
table = sorted(ranking1.iteritems(), key = operator.itemgetter(1), reverse = True)
for row in table:
    print '%-30s %g' % (row[0], row[1])

print 
print 'Ranking2:'
print '--------'
table = sorted(ranking2.iteritems(), key = operator.itemgetter(1), reverse = False)
for row in table:
    print '%-30s %g' % (row[0], row[1])

