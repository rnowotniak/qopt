#!/usr/bin/python

import time
import random
import sys
from multiprocessing import Pool

import numpy as np

DATA = np.matrix('0 7.3 0 0')

def someFun(n):
    np.random.seed(n)
    res = np.random.rand(1,4)
    res[0,0] = n
    return res

pool = Pool(4)

m = np.matrix(np.vstack(pool.map(someFun, [0, 1, 2, 3])))

print m


