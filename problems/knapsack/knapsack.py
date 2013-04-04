#!/usr/bin/python
# 0-1 knapsack problem

import random
import sys
import getopt

def dec2bin(dec, length = None):
    """convert decimal value to binary string"""
    result = ''
    if dec < 0:
        raise ValueError, "Must be a positive integer"
    if dec == 0:
        result = '0'
        if length != None:
            result = result.rjust(length, '0')
        return result
    while dec > 0:
        result = str(dec % 2) + result
        dec = dec >> 1
    if length != None:
        result = result.rjust(length, '0')
    return result


class Knapsack:

    def __init__(self):
        self.items = []
        self.capacity = None

    def generate(self, N):
        # correlated prices
        for i in xrange(N):
            w=random.uniform(1,10)
            p=w+5
            self.items.append((w,p))

        # calculate the capacity as a half of items weight sum
        self.capacity = 0
        for i in xrange(len(self.items)):
            self.capacity += self.items[i][0]
        self.capacity /= 2

    def read(self, filename):
        # read data from the file
        f=open(filename)
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith('#'):
                continue
            if line == '':
                continue
            if self.capacity == None:
                self.capacity = float(line)
                continue
            (w,p) = [float(x) for x in line.split()]
            self.items.append((w,p))
        f.close()

    def write(self, filename):
        f = open(filename, 'w')
        f.write('# capacity\n')
        f.write('%f\n' % self.capacity)
        f.write('\n')

        f.write('# weight price\n')
        for i in xrange(len(self.items)):
            f.write('%f %f\n' % (self.items[i][0],self.items[i][1]))
        f.close()

    def eval(self, k):
        # if not self.items and self.file:
        #     self.read(self.file)

        # calculate price and weight of the knapsack k
        w = 0
        p = 0
        for i in xrange(len(k)):
            if k[i] != '0':
                w += self.items[i][0]
                p += self.items[i][1]
        return (w,p)

    def analyse(self):
        N = len(self.items)
        # iterate over all the possibilite knapsacks of N elements
        for i in xrange(2**N):
            k = dec2bin(i,N)
            e = self.eval(k)
            print k,
            print '  ',
            print e,
            if e[0] > self.capacity:
                print '-'
            else:
                print

class Evaluator:
    def __init__(self):
        self.knapsack = None

    def __call__(self,chrom):
        if not self.knapsack:
            self.knapsack = Knapsack()
            self.knapsack.read(self.file)
            self.ro = None
            for i in self.knapsack.items:
                if not self.ro or self.ro < i[1]/i[0]:
                    self.ro = i[1]/i[0]
        e = self.knapsack.eval(chrom)
        return e[1] - self.ro * (e[0] - self.knapsack.capacity)

def usage():
    print " -g | --generate          -- generate a random knapsack data"
    print " -N | --items <N>         -- number of items"
    print " -f | --file <filename>   -- data filename"
    print " -a | --analyse           -- perform an analysis on the knapsack"

if __name__ == '__main__':
    filename='/dev/stdout'
    generate = False
    analyse = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], "gN:f:a", ['generate', "items=", 'file=', 'analyse'])
    except getopt.GetoptError, err:
        print str(err)
        usage()
        sys.exit(2)
    for o, a in opts:
        if o in ("-N", "--items"):
            N = int(a)
        elif o in ("-f", "--file"):
            filename = a
        elif o in ("-a", "--analyse"):
            analyse = True
        elif o in ("-g", "--generate"):
            generate = True
        else:
            usage()
            assert False, "unhandled option"

    if generate:
        k = Knapsack()
        k.generate(N)
        k.write(filename)
        sys.exit(0)

    if analyse:
        k = Knapsack()
        k.read(filename)
        k.analyse()
        sys.exit(0)

    k = Knapsack()
    k.read(filename)
    print k.capacity
    print k.items
    print len(k.items)
    print k.eval('11')
    k.analyse()

