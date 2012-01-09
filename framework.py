#!/usr/bin/python
#
# Glowny kod frameworka
#

import random
import copy
import math
import time
import sys
import os
import re
import numpy
import traceback,inspect,re
import StringIO
import tempfile
import subprocess


PRNGseed = None

id = '$Id$'
version = '0.' + re.sub(r'[^0-9]', '', '$Revision$')

class Individual:
    def __init__(self):
        self.genotype = None
        self.phenotype = None
        self.fitness = None
        self.cet = None
        # self.xsite
    
    def __str__(self):
        return '(%s, %s, %g)' % (str(self.genotype), str(self.phenotype), float(self.fitness))


class OptAlgorithm:   # TODO:  integrate this with EA  (simplify)

    def __init__(self):
        print 'OptAlgorithm constructor'
        self.maxiter = 20
        self.evaluator = None
        self.best = None # the best individual ever found
        self._time0 = None # timestamp at the start of algorithm
        self.evolutiondata = []

    def initialize(self):
        pass

    def run(self):
        self.initialize()
        self._time0 = time.time()
        #if self.initpopfile:
        #    self.population = load(self.initpopfile)
        self.iter = 0
        while self.iter < self.maxiter:
            self.iter += 1
            print 'iter ' + str(self.iter)
            self.step()

        print '## SOLUTION:'
        print str(self.best)

    def step(self):
        pass

    # allows to pickle objects of this class
    def __getstate__(self):
        result = self.__dict__.copy()
        if result['evaluator']:
            result['evaluator'].__call__ = None
        return result  

class EA(OptAlgorithm):
    def __init__(self):
        OptAlgorithm.__init__(self)
        print 'EA constructor'
        self.popsize = 10
        self.population = []

    def step(self):
        self.evaluation()
        if not self.best:
            self.best = copy.deepcopy(max(self.population, key=lambda ind:ind.fitness)) # minmax issue
        else:
            b = max(self.population, key=lambda ind:ind.fitness) # minmax issue
            if b.fitness > self.best.fitness: # minmax issue
                self.best = copy.deepcopy(b)
        Logging.stat(self)
        print self
        self.operators()

    def initialize(self):
        pass

    def evaluation(self):
        print 'EA evaluating'
        for ind in self.population:
            res = self.evaluator(ind.genotype)
            if type(res) == tuple:
                ind.fitness, ind.phenotype = res
            else:
                ind.fitness, ind.phenotype = res, None

    def operators(self):
        pass

    def __str__(self):
        if not self.population:
            return '(empty population)'
        return '\n'.join([str(ind) for ind in self.population])


class GA(EA):
    def __init__(self):
        EA.__init__(self)
        print 'GA constructor'
        self.chromlen = 10


class ExecutableAlgorithm(OptAlgorithm):
    """The algorithm implemented in external executable file"""
    def __init__(self, *args):
        OptAlgorithm.__init__(self)
        self.best = Individual()
        self.maxiter = 130
        self.proc=subprocess.Popen(
                 args,
                shell=False,stdout=subprocess.PIPE, close_fds=True)
    def run(self):
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            print line,


# Odpowiedzialnoscia tej klasy jest tworzenie pliku z logiem (w tym naszym ustalonym formacie).
# Ta klasa ma zastepowac strumien sys.stdout.
# Na __stdout__  kopiuje wszystko bez ruszania czegokolwiek.
# Wszystkie linie zaczynaja sie od #, oprocz linii STAT.
class Logging:

    statregexp = re.compile(r'^.*\bSTAT\s+([\w.+-]+\s+){6}[\w.+-]+')

    def __init__(self, logfilename, algo=None):
        self.logfilename = logfilename
        self.logfile = open(self.logfilename, 'w+')
        self.newline = True # indicates if the last line printed ends with the new line character
        self.debugmode = False
        sys.stdout = self
        if algo:
            # log the preamble
            self.header(algo)

    def write(self, msg):
        sys.__stdout__.write(msg) # copy msg to __stdout__

        # settle the format of the log line
        if self.debugmode:
            frame = traceback.extract_stack()[-2]
            prefix = '#%s:%d# ' % (frame[0].split('/')[-1], frame[1])
        elif self.newline and msg.startswith('#'):
            prefix = '#'
        else:
            prefix = '# '

        if self.newline:
            self.logfile.write(prefix)
        self.logfile.write(re.sub('\n(?=[^$])', '\n'+prefix, msg))

        self.logfile.flush()
        self.newline = msg.endswith('\n')

    def close(self):
        Logging.postprocess(self.logfilename)
        if sys.stdout == self:
            sys.stdout = sys.__stdout__

    def __getattr__(self, name):
        return sys.__stdout__.__getattr__(name)

    def header(self,algo):
        print '# Numerical Experiment'
        print '%s' % time.strftime('%Y-%m-%d %H:%M:%S %Z')
        print ''
        print '# DESCRIPTION:'
        print '%s\n' % algo.__doc__
        print '# COMMAND:'
        cmd = ''
        for a in sys.argv:
            if a.count(' '):
                a = "'%s'" % a
            cmd += a + ' '
        print '%s\n' % cmd
        print '# PARAMETERS:'
        print 'algorithm = %s' % (algo.__class__)
        print 'evaluator = %s' % (algo.evaluator.__class__)
        print 'PRNGseed = %f' % PRNGseed
        print
        for c in (algo, algo.evaluator):
            for param in dir(c):
                if param.startswith('__'):
                    continue
                if param in ('population'):
                    # it is not an attribute
                    continue
                if type(getattr(c,param)) not in (int,float,str,type(lambda x:x)):
                    continue
                print '%s.%s = %s' % (c.__class__.__name__, param, getattr(c,param))
            print
        print
        print '# DEBUG:'
        self.debugmode = True

    @staticmethod
    def stat(algo):
        fs = map(lambda ind: ind.fitness, algo.population)
        best = max(fs)  # minmax issue
        worst = min(fs)  # minmax issue
        avg = sum(fs) / len(algo.population)
        stddev = numpy.std(fs)
        print 'STAT  %d %d  %.03f   %g  %g  %g  %g' % \
                (algo.iter, \
                algo.evaluator.evaluationCounter,\
                (time.time() - algo._time0)*1000, \
                best, avg, worst, stddev)


    # log nie moze byc wygenerowany od razu w pozadanym formacie w jednym przebiegu.
    # po wygenerowaniu logu musi byc postprocessing.
    # Odpowiedzialnosc tej funkcji: utworzenie sekcji DATA, przeniesienie sekcji SOLUTION
    @staticmethod
    def postprocess(filename):
        stats = StringIO.StringIO()
        solution = StringIO.StringIO()

        # read the logfile and store the data
        logfile = open(filename, 'r')
        while True:
            line = logfile.readline()
            if not line:
                break
            if re.match(r'^## DATA:', line):
                raise Exception('%s is an already postprocessed file' % filename)
            if Logging.statregexp.match(line):
                stats.write(re.sub(r'^.*\bSTAT\b\s+', '', line))
            if re.match(r'.*## SOLUTION:$', line.strip()):
                solution.write(''.join( \
                        [re.sub(r'^#[^ #]+# ', '# ', l) for l in logfile.readlines()]))
                solution.write(''.join(logfile.readlines()))
                break

        # read the logfile again and write the result file
        logfile.seek(0)
        tmpfd, tmpname = tempfile.mkstemp()
        f2 = open(tmpname, 'w')
        while True:
            line = logfile.readline()
            if not line:
                break
            if line.startswith('## DEBUG:'):
                f2.write('## DATA:\n')
                f2.write('# iter     evals   time    best   avg   worst  std_dev\n')
                f2.write(stats.getvalue())
                f2.write('\n')
                f2.write('## SOLUTION:\n')
                f2.write(solution.getvalue())
                f2.write('\n')
            f2.write(line)
        f2.close()
        stats.close()
        solution.close()
        os.rename(tmpname, filename)



# To nie jest juz core framework -- ponizsze CHYBA nalezy przeniesc do jakiegos operators.py
# **   OPERATORY   **

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




