#!/usr/bin/python
# encoding=utf-8

# http://www.scipy.org/Cookbook/Matplotlib/AdjustingImageSize

import sys, numpy
import qopt, qopt.problems
import pylab
import matplotlib.cm
from math import cos, sin

from qopt.framework import matches

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



import matplotlib

###  DOES NOT WORK  ##########################
#      #matplotlib.rc('text', usetex=True)
#      #matplotlib.rc('text', dvipnghack=True)
#      matplotlib.rcParams['text.latex.preamble']=[
#              r'''\usepackage{tikz}\usepackage{graphics}
#              
#      \newcommand{\strzalka}[1]{%
#      \begin{tikzpicture}[rotate=#1,>=stealth]
#      %\fill[lightgray] (0,0) circle (1ex);
#      \draw[black,ultra thin] (0,0) circle (1ex);
#      \draw[->,line width=1.2] (-1ex,0) -- (1ex,0);
#      \end{tikzpicture}}       '''
#      ]
#      matplotlib.rc('text.latex', preamble='\\usepackage{graphicx}')
##############################################

#schema = '10100***************'
schema = '01*011*********'
schema = '10100**********'
schema = '1*1001*********'
schema = '1010***********'
schema = '10010**********'
#schema = '1000*1*********'
#schema = '1000***********'

schema = '01001**********'
schema = '01*01**'
schema = '*1*0***'

coverage=[]
X = pylab.linspace(0, 200, 200)

for i in xrange(2**len(schema)):
    chromo = qopt.int2bin(i, len(schema))
    coverage.append(-int(matches(chromo, schema)))

#angles=[0, 60, 45, 45, 45, 45, 45, 45]
#angles=[0, 90, 30, 45, 45, 45, 45, 45]
#angles=[45, 45, 30, 30, 45, 70, 45, 45]
#angles=[0, 90, 45, 45]

#angles=[(0,1,0,.5), (0.8,.4,.8,.2), (.5,.6,.7,.8)]

#angles=[0, 90, 30, 0, 90, 90,45,45,45,45,45,45,45,45,45]
#angles=[0, 90, 30, 30, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45]
#angles=[45, 45, 25, 10, 45, 45, 30, 45, 45, 45, 45, 45, 45, 45, 45]
#angles=[0, 45,45,45,45,45,45,45,45,45,45,45,45,45,45]
#angles=[0,0,0,0]

#angles=[0, 90, 0, 45]
#angles=[20, 90, 45, 30]
angles=[30, 80, 35, 35]

coverage=[]

angles = [1.*a*numpy.pi / 180 for a in angles]
for i in xrange(2**len(angles)):
    bstr = qopt.int2bin(i,len(angles))
    #print bstr
    p = 1.
    for j in xrange(len(bstr)):
        if bstr[j] == '0':
            p *= numpy.cos(angles[j])# ** 2
        else:
            p *= numpy.sin(angles[j])# ** 2
    coverage.append(-p)
    #print i, p

#coverage=[]
#for i in xrange(4**len(angles)):
#    print i
#    bstr = qopt.dec2four(i,3)
#    print bstr
#    p = 1.
#    for j in xrange(len(bstr)):
#        if bstr[j] == '0':
#            p *= angles[j][0]
#        elif bstr[j] == '1':
#            p *= angles[j][1]
#        elif bstr[j] == '2':
#            p *= angles[j][2]
#        elif bstr[j] == '3':
#            p *= angles[j][3]
#    coverage.append(-p)
#    #print i, p

pylab.plot(X, [f1.evaluate2(x) for x in X], label='Funkcja $f_1(x)$')
#points = numpy.array([15.5,62,140,178])
#pylab.plot(points, [f1.evaluate2(x) for x in points], 'ro', markersize=10)
pylab.xticks((0, len(X)/4,len(X)/2, len(X)/4*3., len(X)),
        ('000...0', '010...0', '100...0', '110...0', '111...1'))
#qgen = []
#def gauss2d(x, shift, sigma):
        #return 1./sigma/numpy.sqrt(2*numpy.pi) * numpy.exp(-((x-shift)**2)/2/sigma**2)
#for q in X:
    #qgen.append(-gauss2d(q, 60, 4) - gauss2d(q, 80, 5) - gauss2d(q, 150, 7) - gauss2d(q, 18, 6))
#pylab.imshow(numpy.matrix(qgen), interpolation='none',extent=(0,200,-10, 100), cmap = matplotlib.cm.gray, alpha=.35)
#pylab.imshow(numpy.matrix([-f1.evaluate(x) for x in X]), interpolation='none',extent=(0,200,-10, 100), cmap = matplotlib.cm.gray, alpha=.35)
#pylab.imshow(numpy.matrix(numpy.random.rand(2**5)), interpolation='none',extent=(0,200,-10, 100), cmap = matplotlib.cm.gray, alpha=.35)
pylab.imshow(numpy.matrix(coverage), interpolation='none',extent=(0,200,0,100), cmap = matplotlib.cm.gray, alpha=.3)
pylab.ylim([0,100])
pylab.xlim((0,200))
pylab.grid(True)
#pylab.legend(loc='upper left')
#pylab.title('$\\left[' +'|'.join(['{{{%.3f \\atop %.3f} \\atop %.3f} \\atop %.3f}' % (a[0],a[1],a[2],a[3]) for a in angles]) + '\\right]$')

pylab.title(r'\(q='+ \
        ''.join([
            r'\left[\begin{array}{c}%.3f\\%.3f\end{array}\right]' % (cos(angle), sin(angle))
            for angle in angles]) + r'\)'
)

#pylab.title('''q=\\rotatebox{45}{\includegraphics[width=1.5ex]{/tmp/strzalka.eps}}''')  # DOES NOT NOT NOT WORK !

pylab.xlabel('$x$')
pylab.ylabel('$f(x)$')
#pylab.gca().set_aspect(1, 'box')
pylab.savefig('/tmp/f1.pdf', dpi=300, bbox_inches = 'tight')
pylab.cla()


############################
sys.exit(0)
############################





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

