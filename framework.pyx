#!/usr/bin/python
#
# QOpt framework
# Copyright (C) 2012   Robert Nowotniak
#

import random
import copy
import math
import time
import sys
import os
import re
import subprocess
import numpy as np

cimport libc.string

cdef extern from "framework.h":
    int c_matches "matches" (char *chromo, char *schema, int length)
    void c_dec2four "dec2four"(char *buf, long int dec, int length)

def dec2four(long int dec, int length):
    buf = '_' * length
    c_dec2four(buf, dec, length)
    return buf

def matches(char *chromo, char *schema):
    length = libc.string.strlen(chromo)
    return bool(c_matches(chromo, schema, length))

PRNGseed = None

def int2bin(n, count=24):
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

# XXX Move it somewhere else!!
def M(qchromo,char *schema):
    cdef float result = 1.
    for i in xrange(len(schema)):
        if schema[i] == '0':
            result *= np.square(np.cos(qchromo[i]))
        elif schema[i] == '1':
            result *= np.square(np.sin(qchromo[i]))
    return result

def E(Q,schema):
    sum1 = 0.
    cdef float elem
    for w in xrange(len(Q)+1):
        sum2 = 0.
        for c in xrange(2**len(Q)):
            bstr = int2bin(c, len(Q))
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
            bstr = int2bin(c, len(Q))
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




#
# Klasa pomocnicza, w ktorej mozna agregowac informacje o osobniku,
# np. individual1.fitness, individual1.genotype, individual1.whatever
#
class Individual:
    def __init__(self, **kwargs):
        self.genotype = None
        self.fitness = None
        # self.phenotype = None
        # self.cet = None
        # ...
        for k in kwargs:
            setattr(self, k, kwargs[k])
    
    def __str__(self):
        return '(%s, %g)' % (str(self.genotype), float(self.fitness))


class EA:
    def __init__(self):
        print 'EA __init__'
        self.t = 0

    def run(self):
        self.t = 0
        self.initialize()
        while self.t < self.tmax:
            self.t += 1
            self.generation()


#   #
#   # Moze dorobic taka ciekawa rzecz: XXX
#   #   .history (?)
#   #   ta magiczna tablica zawieralaby stan zmiennych z kolejnych generacji algorytmu,
#   #   tzn. moze byloby sie po wykonaniu algorytmu odwolywac np:
#   #   alg1.history[nrgen].best  albo  alg1.history[nrgen].Q   (ile RAM-u by to bralo?)
#   #   nalezaloby to uwzglednic w .step
#   #   Mozna byloby tez po calym eksperymencie latwo zapisywac stan i go potem latwo debugowac,
#   #   tzn. np. robic pickle(self, '/tmp/blaa.id')
#   #
#   #
#   class EA:
#       """ The base class for evolutionary algorithms (population-based heuritics) """
#   
#       def __init__(self):
#           self.popsize = 10
#           self.population = []
#   
#           self.best = None
#           """ the best individual ever found (Individual object or its genotype directly) """
#   
#           self.t = 0 # generation number
#   
#           # callbacks
#           self.stepCallback = None
#           self.evaluator = None
#           self.evaluation_counter = 0
#   
#           # minmaxop -> x IS BETTER THAN y
#           self.minmax = min
#           self.minmaxop = lambda x,y: [lambda x,y: x > y, lambda x,y: x < y][self.minmax == min](x,y)
#   
#           self.__time0 = None # timestamp at the start of algorithm
#   
#           self.tmax = None
#           self.maxNoFE = None
#           self.history = []
#   
#       def initialize(self):
#           pass # abstract
#   
#       def operators(self):
#           pass # abstract
#   
#       def evaluate(self, population):
#           # print 'EA evaluating'
#           results = []
#           for ind in population:
#               res = self.evaluator(ind)
#               results.append(res)
#           self.evaluation_counter += len(population)
#           return results
#   
#   
#       def termination(self):
#           # General termination conditions. This method can be overriden
#           if hasattr(self, 'tmax') and self.tmax is not None:
#               return self.t >= self.tmax
#           if hasattr(self, 'maxNoFE') and self.maxNoFE is not None:
#               return self.evaluation_counter >= self.maxNoFE
#           assert False, 'No termination conditions given'
#   
#   
#       def run(self):
#           # konwencja numerowania generacji:
#           # na etapie inicjalizacji: generacja zerowa,
#           # przy rozpoczeciu pierwszej generacji: generacja pierwsza
#           # (inkrementacja na poczatku tej glownej petli)
#           self.t = 0
#           self.initialize()
#           self.__time0 = time.time()
#           while not self.termination():
#               self.t += 1
#               if self.stepCallback != None:
#                   self.stepCallback(self)
#               # print 't ' + str(self.t)
#               self.generation()
#               #self.__save_history()
#   
#   
#       def generation(self): # this function is volatile to minmax issue
#           """ this function should be overriden in whole by complex EA algorithms """
#           # evaluate
#           fvalues = self.evaluate(self.population)
#           for i in xrange(len(self.population)):
#               self.population[i].fitness = fvalues[i]
#           # store the best solution
#           index_of_best = fvalues.index(self.minmax(fvalues))
#           if self.best == None or self.minmaxop(self.population[index_of_best].fitness, self.best.fitness):
#               self.best = copy.deepcopy(self.population[index_of_best])
#           # operators
#           self.operators()
#   
#   
#       def __save_history(self):
#           copy1 = copy.deepcopy(self)
#           del copy1.history
#           self.history.append(copy1)
#   
#       # -- to byc moze bedzie potrzebne --
#       # allows to pickle objects of this class
#       # def __getstate__(self):
#       #     result = self.__dict__.copy()
#       #     if result['evaluator']:
#       #         result['evaluator'].__call__ = None
#       #     return result  
#   
#       def __str__(self):
#           if not self.population:
#               return '(empty population)'
#           return '\n'.join([str(ind) for ind in self.population])
#   
#   
#   class GA(EA):
#       def __init__(self):
#           EA.__init__(self)
#           print 'GA constructor'
#           self.chromlen = 10
#   
#   
#   class ExecutableAlgorithm(EA):
#       """The algorithm implemented in external executable file"""
#       def __init__(self, *args):
#           EA.__init__(self)
#           self.best = Individual()
#           self.tmax = 130
#           self.proc=subprocess.Popen(
#                    args,
#                   shell=False,stdout=subprocess.PIPE, close_fds=True)
#       def run(self):
#           while True:
#               line = self.proc.stdout.readline()
#               if not line:
#                   break
#               print line,
#   
#   







# --------8<--------8<--------8<--------8<--------8<--------8<--------8<
# To nie jest juz core framework -- ponizsze CHYBA nalezy przeniesc do jakiegos operators.py
# **   OPERATORY   **
# --------8<--------8<--------8<--------8<--------8<--------8<--------8<

def RouletteSelection(population):
    # TODO: zle dziala dla ujemnych
    sects = [ind.fitness for ind in population]
    m = min(sects)
    if m < 0:
        sects = [s - m for s in sects]
    s = sum(sects)
    if s == 0:
        print 'stagnation'
        return population
    sects = [float(i)/s for i in sects]

    # accumulated
    for i in xrange(1, len(population)):
        sects[i] = sects[i - 1] + sects[i]

    #print population
    #print sects
    newpop = []
    for i in xrange(len(population)):
        r = random.random()
        for j in xrange(len(sects)):
            if r <= sects[j]:
                newpop.append(copy.deepcopy(population[j]))
                break

    return newpop

def OnePointCrossover(population, Pc):
    toCrossover = []
    for n in xrange(len(population)):
        if random.random() <= Pc:
            toCrossover.append(n)
    if len(toCrossover) % 2 != 0:
        n = int(math.floor(random.random() * len(population)))
        while toCrossover.count(n) > 0:
            n = int(math.floor(random.random() * len(population)))
        toCrossover.append(n)

    #print toCrossover

    done = []
    for n in xrange(len(toCrossover)):
        par1 = toCrossover[n]
        if done.count(par1) > 0:
            continue

        while True:
            par2 = random.choice(toCrossover)
            if done.count(par2) == 0:
                break

        cp = int(math.floor(random.random() * (len(population[par1].genotype) - 1)))
        # print par1, par2, cp, len(population)
        # print population[par1]
        # print population[par2]
        child1 = population[par1].genotype[:cp] + population[par2].genotype[cp:]
        child2 = population[par2].genotype[:cp] + population[par1].genotype[cp:]
        # print child1
        # print child2
        population[par1].genotype = child1
        population[par2].genotype = child2
        done.append(par1)
        done.append(par2)
        # print population
    return population

def OnePointMutation(population, Pm):
    for n in xrange(len(population)):
        for locus in xrange(len(population[n].genotype)):
            if random.random() <= Pm:
                chrom = list(population[n].genotype)
                if chrom[locus] == '1':
                    chrom[locus] = '0'
                else:
                    chrom[locus] = '1'
                population[n].genotype = ''.join(chrom)
    return population




