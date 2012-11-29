#!/usr/bin/python
# encoding=utf-8

# http://www.scipy.org/Cookbook/Matplotlib/AdjustingImageSize

import sys, numpy
import qopt, qopt.problems
import pylab

#pylab.rc('text', usetex=True)
#pylab.rc('text.latex', unicode=True)

import qopt.problems._sat
s1 = qopt.problems._sat.SatProblem(qopt.path('problems/sat/flat30-100.cnf'))
s2 = qopt.problems._sat.SatProblem(qopt.path('problems/sat/qg4-08.cnf'))
s3 = qopt.problems._sat.SatProblem(qopt.path('problems/sat/hanoi4.cnf'))

k10 = qopt.problems.knapsack10
k15 = qopt.problems._knapsack.KnapsackProblem(qopt.path('problems/knapsack/knapsack-15.txt'))
k20 = qopt.problems._knapsack.KnapsackProblem(qopt.path('problems/knapsack/knapsack-20.txt'))
k25 = qopt.problems._knapsack.KnapsackProblem(qopt.path('problems/knapsack/knapsack-25.txt'))
k100 = qopt.problems.knapsack100
k250 = qopt.problems.knapsack250
k500 = qopt.problems.knapsack500

import itertools

def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.izip_longest(fillvalue=fillvalue, *args)

def knapsackplot(prob, length, fname):
    Y = []
    Y2 = []
    for i in xrange(2**length):
        kstr = qopt.int2bin(i, length) # + '1' * (length - 20)
        #prob.repair(kstr)
        #print kstr
        e = prob.evaluate2(kstr)
        #print e
        # e = prob.evaluate(kstr)
        if True: # e[0] <= prob.capacity:
            Y.append(e[1])
            Y2.append(e[0])
    Yg = []
    for g in grouper(2**length/1024, Y):
        Yg.append(max(g))
    Y = Yg
    Y2g = []
    for g in grouper(2**length/1024, Y2):
        Y2g.append(numpy.average(g))
    Y2 = Y2g
    fig1 = pylab.figure()
    ax1 = fig1.add_subplot(111)
    # s = sorted(Y)
    # print s[-10:]
    price = ax1.plot(Y, '-')
    pylab.xlim([0,len(Y)])
    #pylab.ylim([-50,130])
    #pylab.ylim([520,521])
    pylab.ylabel(u'Wartość')
    ax1.grid(True)
    ax2 = fig1.add_subplot(111, sharex=ax1, frameon = False)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    weight = ax2.plot(Y2, 'r')
    ax2.plot([0,len(Y2)], [prob.capacity, prob.capacity], 'r')
    pylab.ylim([min(Y2),max(Y)])
    pylab.xlim([0,len(Y)])
    pylab.xticks((0, len(Y)/4,len(Y)/2, len(Y)/4*3., len(Y)),
            ('000...0', '010...0', '100...0', '110...0', '111...1'))
    pylab.ylabel(u'Waga')
    pylab.xlabel(u'Przestrzeń rozwiązań $X$')
    pylab.legend((price[0],weight[0]), (u'Wartość','Waga'), loc='upper left', shadow=True, fancybox=True, borderpad=0.3)
    pylab.savefig(fname, bbox_inches = 'tight')


def satplot(prob, length, fname):
    global Y
    Y = []
    for i in xrange(2**length):
        kstr = qopt.int2bin(i, length) # + '0' * (length - 14)
        Y.append(prob.evaluate(kstr))
    Yg = []
    for g in grouper(2**length/1024, Y):
        Yg.append(max(g))
    Y = Yg
    pylab.figure()
    pylab.plot(Y)
    pylab.xlim([0,len(Y)])
    pylab.grid(True)
    pylab.yticks(numpy.arange(pylab.ylim()[0] + 1, pylab.ylim()[1] + 1))
    pylab.xticks((0, len(Y)/4,len(Y)/2, len(Y)/4*3., len(Y)),
            ('000...0', '010...0', '100...0', '110...0', '111...1'))
    pylab.ylabel(u'Liczba prawdziwych klauzul')
    pylab.xlabel(u'Przestrzeń rozwiązań $X$')
    pylab.savefig(fname, bbox_inches = 'tight')

#knapsackplot(k15, 15, '/tmp/knapsack15.pdf')
#knapsackplot(k20, 20, '/tmp/knapsack20.pdf')
#knapsackplot(k25, 25, '/tmp/knapsack25.pdf')

#  rs1 = qopt.problems._sat.SatProblem(qopt.path('problems/sat/random-15.cnf'))
#  rs2 = qopt.problems._sat.SatProblem(qopt.path('problems/sat/random-20.cnf'))
#  rs3 = qopt.problems._sat.SatProblem(qopt.path('problems/sat/random-25.cnf'))
#  #rs3 = qopt.problems._sat.SatProblem(qopt.path('contrib/SPY-1.2/25.cnf'))
#  #satplot(rs1, 15, '/tmp/sat1.pdf')
#  print 'a'
#  #satplot(rs2, 20, '/tmp/sat2.pdf')
#  print 'a'
#  satplot(rs3, 25, '/tmp/sat3.pdf')
#  print 'a'
#  
#  sys.exit(0)
#  
#  # satplot(s1, 90, '/tmp/sat1.pdf')
#  # satplot(s2, 512, '/tmp/sat2.pdf')
#  # satplot(s3, 718, '/tmp/sat3.pdf')
#  
#  sys.exit(0)


f1 = qopt.problems.func1d.f1
f2 = qopt.problems.func1d.f2
f3 = qopt.problems.func1d.f3

pylab.figure()

schema = '01*01**********'
schema = '01001**********'
schema = '01*011*********'
schema = '010*11*********'
schema = '01*01**********'
schema = '*10011*********'
schema = '01*010*********'

#   # sat 15
#   schema = '****001*11*****'
#   
def matches(chromo, schema):
    for i in xrange(len(chromo)):
        if schema[i] == '*':
            continue
        if schema[i] != chromo[i]:
            return False
    return True
#   
#   coverage = []
#   
#   rs1 = qopt.problems._sat.SatProblem(qopt.path('problems/sat/random-15.cnf'))
#   satplot(rs1, 15, '/tmp/sat1.pdf')
#   for i in xrange(2**len(schema)):
#       chromo = qopt.int2bin(i, len(schema))
#       coverage.append(((200*int(matches(chromo, schema)) - 50)))
#   pylab.fill_between(pylab.linspace(0,len(Y),len(coverage)), -50, coverage, alpha = .2)
#   pylab.ylim([56,68])
#   pylab.savefig('/tmp/sat1.pdf', bbox_inches = 'tight')
#   sys.exit(0)



schema = '10100***************'

coverage=[]
X = pylab.linspace(-5, 5, 200)
pylab.plot(X, [f2.evaluate(x) for x in X], label='Funkcja $f_2(x)$')
for i in xrange(2**len(schema)):
    chromo = qopt.int2bin(i, len(schema))
    coverage.append(((200*int(matches(chromo, schema)) - 50)))
pylab.fill_between(pylab.linspace(0,len(X),len(coverage)), -50, coverage, alpha = .2)
pylab.ylim([-3,4])
pylab.xlim((-5,5))
pylab.grid(True)
pylab.legend(loc='upper left')
pylab.xlabel('$x$')
pylab.ylabel('$f_2(x)$')
#pylab.gca().set_aspect(1, 'box')
pylab.savefig('/tmp/f2.pdf', bbox_inches = 'tight')
pylab.cla()
sys.exit(0)

X = pylab.linspace(0, 200, 200)
pylab.plot(
        [0, 10, 20, 30, 40, 56, 60, 65, 80, 90, 100, 120, 150, 180, 200],
        [0, 20, 40, 10, 25, 33, 80, 45, 60, 20, 0, 20, 40, 20, 0],
        'ro', markersize = 10, label=u'Węzły interpolacji')
Y = [f1.evaluate(x) for x in X]
pylab.plot(X, Y, label=u'Funkcja interpolująca $f_1$')
for i in xrange(2**len(schema)):
    chromo = qopt.int2bin(i, len(schema))
    coverage.append(((200*int(matches(chromo, schema)) - 50)))
#pylab.fill_between(pylab.linspace(0,200,len(coverage)), 0, coverage, '-')
pylab.fill_between(pylab.linspace(0,200,len(coverage)), -50, coverage, alpha = .2)
#pylab.xlim((-5,5))
pylab.legend(loc='upper right')
pylab.grid(True)
pylab.xlabel('$x$')
pylab.ylabel('$f_1(x)$')
pylab.gcf().get_size_inches()[0] *= 2
pylab.title(schema)
pylab.xlim([0,200])
pylab.ylim([-10,100])
pylab.savefig('/tmp/f1.pdf', bbox_inches = 'tight')
pylab.cla()

