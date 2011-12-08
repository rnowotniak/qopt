#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import random
import copy

import testfuncs
#import ctestfuncs

import psyco
psyco.full()

# from matplotlib.path import Path
# from matplotlib.patches import PathPatch
# import matplotlib.pyplot as plt
import numpy as np

# returens a copy of sequence seq with no duplicates
def unique(seq): 
    # order preserving
    checked = []
    for e in seq:
        if e not in checked:
            checked.append(e)
    return checked


def drawPDFs(pdfs):
    plt.clf()
    for g in xrange(G):
        pdf = pdfs[g]
        ax = plt.subplot(2, G, g + 1)
        plt.grid(True)
        for i in xrange(len(pdf)):
            plt.barh(0, left = pdf[i][0], width = pdf[i][1], height = pdf[i][2], color='#707070')
        plt.xlim([-30,30])
    #for g in xrange(G):
    #    pdf = pdfs[g]
    #    ax = plt.subplot(2, G, G + g + 1)
    #    plt.grid(True)
    #    for q in Q:
    #        plt.barh(0, left = q[g][0] - q[g][1]/2, width = q[g][1], height = q[g][2], color = 'red', alpha = 0.3)
    #    plt.xlim([-10,20])
    #    #plt.show()
    plt.savefig('/tmp/pdf.pdf', bbox_inches='tight')

# interference / summing square pulses
def getPDFs(qpop):
    pdfs = []
    for g in xrange(G):
        pulses = []
        for i in xrange(len(qpop)):
            # beginning, end, height
            pulse = (qpop[i][g][0] - qpop[i][g][1]/2, qpop[i][g][0] + qpop[i][g][1]/2, qpop[i][g][2])
            pulses.append(pulse)
        points = unique(sorted(map(lambda p: p[0], pulses) + map(lambda p: p[1], pulses)))
        if len(points) == 1: # the pulses have converged completelly already
            # denote this special case as a PDF with (0,+inf)
            pdf = [(pulses[0][0], 0, float('inf'))]
            pdfs.append(pdf)
            continue
        pdf = []
        for i in xrange(len(points) - 1):
            x = points[i]
            h = 0.
            for p in pulses:
                if p[0] <= x and p[1] > x:
                    h += p[2]
            pdf.append((x, points[i + 1] - x, h))
        pdfs.append(pdf)
    return pdfs

def crossover(pop1, pop2):
    # pop1 = [[9.400, -19.846], [8.604, -13.675], ...
    # pop2 = [([9.400, -19.846], 117), ([8.604, -13.675], 19.3), ...
    result = copy.deepcopy(pop1)
    for i in xrange(K):
        for j in xrange(G):
            r = random.random()
            if r < XI:
                pass
            else:
                result[i][j] = pop2[i][0][j]
    return result


def selection(pop, howmany):
    result = sorted(pop, cmp = lambda x,y: cmp(x[1], y[1])) # minmax issue
    return result[:howmany]

evaluation_counter = 0
fitness_function = testfuncs.f3

def evaluate(P):
    global evaluation_counter
    # P = [[9.400, -19.846], [8.604, -13.675], ...
    result = []
    for ind in P:
        evaluation_counter += 1
        fitness = fitness_function(ind)
        result.append((ind, fitness))
    # result = [([9.400, -19.846], 17), ([8.604, -13.675], 19.3), ...
    return result

def observe(pdfs):
    # print '---'
    result = []
    for g in xrange(len(pdfs)):
        pdf = pdfs[g]
        if len(pdf) == 1 and pdf[0][1] == 0:
            # the special case
            x = pdf[0][0]
            result.append(x)
            continue

        r = random.random()
        # print r
        s = 0.
        x = 'XXX'
        for i in xrange(len(pdf)):
            p = pdf[i]
            if s + p[1] * p[2] < r:
                s += p[1] * p[2]
            else:
                a = p[2]
                b = s - a * p[0]
                x = (r - b) / a
                if x < bounds[g][0]:  # FUCK! a mistake.. ;/
                    x = bounds[g][0]
                elif x > bounds[g][1]:
                    x = bounds[g][1]
                break
        if x == 'XXX':
            #print 'area under PDF is not 1\n', pdf # floating point arithmetic issue
            #print 'XXX'
            #print pdf
            x = pdf[-1][0] + random.random() * pdf[-1][1] # little fix
            #sys.exit(0)
        result.append(x)
    return result

popsize = 10
popsize = 2 # for figure
K = 10 # size of classical population
G = 30  # number of genes
G = 3  # for figure
maxiter = 5000
XI = 0.1 # crossover rate
DELTA = 0.5 # contraction factor
kstep = 5
bounds = [(-600,600)]*30
EPSILON = 1e-6

sys.argv = ['', '1', '0.5', '0.5']

if sys.argv[1] == '1':
    fitness_function = testfuncs.f1
    bounds = [(-30,30)]*30
    popsize = 10
    K = 10
    G = 30
    kstep = 10
    XI = float(sys.argv[2])
    DELTA = float(sys.argv[3])

    maxiter = 200
elif sys.argv[1] == '2':
    fitness_function = testfuncs.f2
    bounds = [(-10,10)]*30
    popsize = 5
    K = 5
    G = 30
    kstep = 8
    XI = float(sys.argv[2])
    DELTA = float(sys.argv[3])

    maxiter = 500
elif sys.argv[1] == '3':
    fitness_function = testfuncs.f3
    bounds = [(-600,600)]*30
    #plots were generated for these parameters:
    # XI = 0.1
    #--
    popsize = 5
    K = 10
    G = 30
    kstep = 5
    XI = float(sys.argv[2])
    DELTA = float(sys.argv[3])
elif sys.argv[1] == '4':
    fitness_function = testfuncs.f4
    bounds = [(-10,10)]*30
    popsize = 4
    K = 4
    G = 30
    kstep = 20
    XI = float(sys.argv[2])
    DELTA = float(sys.argv[3])


# initialization
Q = []
for i in xrange(popsize):
    q = []
    for j in xrange(G):
        center = random.random() * (bounds[j][1] - bounds[j][0]) + bounds[j][0]
        width = bounds[j][1] - bounds[j][0]
        height = 1. / width / popsize
        q.append([center,1.*width, height])
    Q.append(q)

#for i in xrange(popsize):
#    print Q[i]
#

#for i in xrange(popsize):
#    for j in xrange(G):
#        ax = plt.subplot(popsize, G, i * G + j + 1)
#        plt.grid(False)
#        ax.barh(bottom = 0, left = Q[i][j][0] - Q[i][j][1]/2, width = Q[i][j][1], height = Q[i][j][2]*popsize, color='#E0E0E0', linewidth=2)
#        ax.set_xlim(bounds[j])
#        ax.set_ylim(0, 0.04)
#        ax.set_yticks([])
#        if i < popsize - 1:
#            ax.set_xticks([])
#plt.savefig('/tmp/qpop.pdf', bbox_inches='tight')


iter = 0
while iter <= maxiter:
    pdfs = getPDFs(Q)
    # pdfs = [ [(x,w,h), (x,w,h), (x,w,h), ...], [(x,w,h), (x,w,h), (x,w,h), ...], ... ]
    #if iter == 35:
    #    drawPDFs(pdfs)
    #    sys.exit(0)

    E = [observe(pdfs) for i in xrange(K)]
    # E = [[9.400, -19.846], [8.604, -13.675], ...

    #E = evaluate(E)
    # E = [([9.400, -19.846], 117), ([8.604, -13.675], 19.3), ...

    # --- proof that the sampling is correct
    # plt.clf()
    # plt.hold(True)
    # for g in xrange(G):
    #     ax = plt.subplot(1, G, g + 1)
    #     for n in xrange(1000):
    #         ax.plot(observe(pdfs)[g], random.random(), '*')
    #         ax.set_xlim((-10,20))
    #         ax.set_ylim((0,1))
    # plt.savefig('/tmp/e.pdf')
    # ---

    if iter == 0:
        oldC = None
        fi = None
        C = E
        C = evaluate(C)
        # C = [([9.400, -19.846], 117), ([8.604, -13.675], 19.3), ...
        C = sorted(C, cmp = lambda x,y: cmp(x[1], y[1])) # minmax issue
        # C = [ ([8.604, -13.675], 19.3), ([9.400, -19.846], 117), ...
    else:
        if iter%kstep == 0: # every kstep-th iteration
            oldC = copy.deepcopy(C)
        E = crossover(E, C)
        E = evaluate(E)
        # E = [([9.400, -19.846], 117), ([8.604, -13.675], 19.3), ...
        C = selection(E + C, K)
        print evaluation_counter, C[0][1]
        sys.stdout.flush()
        #if C[0][1] < EPSILON:
        #    sys.exit(0)
        # C = [ ([8.604, -13.675], 19.3), ([9.400, -19.846], 117), ...
        # C is also already sorted now

    if iter%kstep == 0 and oldC:
        improved = 0
        for i in xrange(len(C)):
            if C[i][1] < oldC[i][1]: # minmax issue
                improved += 1
        fi = 1.0 * improved / len(C)
    # update Q(t)
    for i in xrange(popsize):
        for j in xrange(G):
            Q[i][j][0] = C[i][0][j] # translate center of the i-th pulse
            if iter % kstep or not oldC:
                continue
            # scale the pulse width (update the height also)
            #print Q[0]
            if fi < 0.2:
                Q[i][j][1] *= DELTA
            elif fi > 0.2:
                Q[i][j][1] /= DELTA
            if Q[i][j][1] > 0:
                Q[i][j][2] = 1. / Q[i][j][1] / popsize
        
    iter += 1

#print C[0]

