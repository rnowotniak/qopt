#!/usr/bin/python

import sys
import random
import os


while True:
    xi = random.random()
    delta = random.random()
    os.system('./multiprocessing-iqiea-run.py %g %g' % (xi, delta))

