#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import random
import copy
import math

import qopt.framework

# import psyco
# psyco.full()

# random.seed(3)

# from matplotlib.path import Path
# from matplotlib.patches import PathPatch
#import matplotlib.pyplot as plt
import numpy as np

# TEST FUNCTIONS
def testfuncs_f1(x):
    result = 0.
    for j in xrange(len(x)):
        result += x[j]**2
    return result
def testfuncs_f2(x):
    result = 0.
    for j in xrange(len(x)):
        result += abs(x[j])
    prod = 1.
    for j in xrange(len(x)):
        prod *= abs(x[j])
    result += prod
    return result
def testfuncs_f3(x):
    result = 0.
    for j in xrange(len(x)):
        result += x[j]**2
    result *= 1./4000
    p = 1.
    for j in xrange(len(x)):
        p *= math.cos(x[j]/math.sqrt(j+1))
    result = result - p + 1
    return result
def testfuncs_f4(x):
    result = 0
    for j in xrange(len(x)):
        result += x[j]**2
    result *= 1./len(x)
    result = -20 * math.exp(-0.2 * math.sqrt(result))
    b = 0
    for j in xrange(len(x)):
        b += math.cos(2.*math.pi*x[j])
    b *= 1./len(x)
    result -= math.exp(b)
    result += 20 + math.e
    return result


# returns a copy of sequence seq with no duplicates
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




def selection(pop, howmany):
    result = sorted(pop, cmp = lambda x,y: cmp(x[1], y[1])) # minmax issue
    return result[:howmany]


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


class iQIEA(qopt.framework.EA):

    def __init__(self):
        qopt.framework.EA.__init__(self)
        self.evaluation_counter = 0

        # defaults
        self.popsize = 10
        self.K = 10 # size of classical population
        self.G = 30  # number of genes
        self.maxiter = 5000
        self.XI = 0.1 # crossover rate
        self.DELTA = 0.5 # contraction factor
        self.kstep = 5
        self.bounds = [(-600,600)]*30
        self.EPSILON = 1e-6
        self.evolutiondata = [] # XXX (inherited)

    def initialize(self):
        self.Q = []
        for i in xrange(self.popsize):
            q = []
            for j in xrange(self.G):
                center = random.random() * (self.bounds[j][1] - self.bounds[j][0]) + self.bounds[j][0]
                width = self.bounds[j][1] - self.bounds[j][0]
                height = 1. / width / self.popsize
                q.append([center,1.*width, height])
            self.Q.append(q)
        self.iter = 0
        self.evolutiondata = [] # XXX
        self.evaluation_counter = 0

    def step(self):
        pdfs = self.getPDFs(self.Q)
        # pdfs = [ [(x,w,h), (x,w,h), (x,w,h), ...], [(x,w,h), (x,w,h), (x,w,h), ...], ... ]
        #if self.iter == 35:
        #    drawPDFs(pdfs)
        #    sys.exit(0)

        E = [self.observe(pdfs) for i in xrange(self.K)]
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

        if self.iter == 0:
            oldC = None
            fi = None
            self.C = E
            self.C = self.evaluate(self.C)
            # C = [([9.400, -19.846], 117), ([8.604, -13.675], 19.3), ...
            self.C = sorted(self.C, cmp = lambda x,y: cmp(x[1], y[1])) # minmax issue
            # C = [ ([8.604, -13.675], 19.3), ([9.400, -19.846], 117), ...
        else:
            if self.iter%self.kstep == 0: # every kstep-th iteration
                oldC = copy.deepcopy(self.C)
            E = self.crossover(E, self.C)
            E = self.evaluate(E)
            # E = [([9.400, -19.846], 117), ([8.604, -13.675], 19.3), ...
            self.C = selection(E + self.C, self.K)
            print self.evaluation_counter, self.C[0][1]
            self.best = self.C[0]
            #if self.C[0][1] < EPSILON:
            #    sys.exit(0)
            # C = [ ([8.604, -13.675], 19.3), ([9.400, -19.846], 117), ...
            # C is also already sorted now

        if self.iter%self.kstep == 0 and oldC:
            improved = 0
            for i in xrange(len(self.C)):
                if self.C[i][1] < oldC[i][1]: # minmax issue
                    improved += 1
            fi = 1.0 * improved / len(self.C)
        # update Q(t)
        for i in xrange(self.popsize):
            for j in xrange(self.G):
                self.Q[i][j][0] = self.C[i][0][j] # translate center of the i-th pulse
                if self.iter % self.kstep or not oldC:
                    continue
                # scale the pulse width (update the height also)
                #print Q[0]
                if fi < 0.2:
                    self.Q[i][j][1] *= self.DELTA
                elif fi > 0.2:
                    self.Q[i][j][1] /= self.DELTA
                if self.Q[i][j][1] > 0:
                    self.Q[i][j][2] = 1. / self.Q[i][j][1] / self.popsize
        #self.iter += 1
        self.evolutiondata.append((self.evaluation_counter, self.C[0][1])) # XXX move to step in upper class

        #print C[0]

    # interference / summing square pulses
    def getPDFs(self, qpop):
        pdfs = []
        for g in xrange(self.G):
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

    def observe(self, pdfs):
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
                    if x < self.bounds[g][0]:  # FUCK! a mistake.. ;/
                        x = self.bounds[g][0]
                    elif x > self.bounds[g][1]:
                        x = self.bounds[g][1]
                    break
            if x == 'XXX':
                #print 'area under PDF is not 1\n', pdf # floating point arithmetic issue
                #print 'XXX'
                #print pdf
                x = pdf[-1][0] + random.random() * pdf[-1][1] # little fix
                #sys.exit(0)
            result.append(x)
        return result

    def evaluate(self, P):
        # P = [[9.400, -19.846], [8.604, -13.675], ...
        result = []
        for ind in P:
            self.evaluation_counter += 1
            fitness = self.fitness_function(ind)
            result.append((ind, fitness))
        # result = [([9.400, -19.846], 17), ([8.604, -13.675], 19.3), ...
        return result

    def crossover(self, pop1, pop2):
        # pop1 = [[9.400, -19.846], [8.604, -13.675], ...
        # pop2 = [([9.400, -19.846], 117), ([8.604, -13.675], 19.3), ...
        result = copy.deepcopy(pop1)
        for i in xrange(self.K):
            for j in xrange(self.G):
                r = random.random()
                if r < self.XI:
                    pass
                else:
                    result[i][j] = pop2[i][0][j]
        return result



if __name__ == '__main__':
    iqiea = iQIEA()
    sys.argv = 'rcqiea.py 3 0.78 0.421'.split()
    sys.argv = 'rcqiea.py 3 0.5 0.1'.split()
    sys.argv = 'rcqiea.py 3 0.23 0.785'.split()
    if sys.argv[1] == '1':
        iqiea.fitness_function = testfuncs_f1
        iqiea.bounds = [(-30,30)]*30
        iqiea.popsize = 10
        iqiea.K = 10
        iqiea.G = 30
        iqiea.kstep = 10
        iqiea.XI = float(sys.argv[2])
        iqiea.DELTA = float(sys.argv[3])
        iqiea.maxiter = 200
    elif sys.argv[1] == '2':
        iqiea.fitness_function = testfuncs_f2
        iqiea.bounds = [(-10,10)]*30
        iqiea.popsize = 5
        iqiea.K = 5
        iqiea.G = 30
        iqiea.kstep = 8
        iqiea.XI = float(sys.argv[2])
        iqiea.DELTA = float(sys.argv[3])
        iqiea.maxiter = 500
    elif sys.argv[1] == '3':
        iqiea.fitness_function = testfuncs_f3
        iqiea.bounds = [(-600,600)]*30
        iqiea.popsize = 10
        iqiea.K = 10
        iqiea.G = 30
        iqiea.kstep = 5
        iqiea.XI = float(sys.argv[2])
        iqiea.DELTA = float(sys.argv[3])
        iqiea.maxiter = 800
    elif sys.argv[1] == '4':
        iqiea.fitness_function = testfuncs_f4
        iqiea.bounds = [(-10,10)]*30
        iqiea.popsize = 4
        iqiea.K = 4
        iqiea.G = 30
        iqiea.kstep = 20
        iqiea.XI = float(sys.argv[2])
        iqiea.DELTA = float(sys.argv[3])
    iqiea.run()


