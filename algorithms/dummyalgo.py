#!/usr/bin/python

import sys
import random
import time
import os

val = 5000

for evals in xrange(100000):
    val *= .99995 + (random.random() * .000015)
    time.sleep(0.00001)
    print evals, val, os.getpid()



