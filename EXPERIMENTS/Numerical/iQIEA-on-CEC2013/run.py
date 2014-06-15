#!/usr/bin/python

import sys
import random
import os


while True:
    xi = random.random()
    delta = .75 + random.random() * .25
    os.system('./multiprocessing-iqiea-run.py %g %g' % (xi, delta))

