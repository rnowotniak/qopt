#!/usr/bin/python

import sys
import time,random

#time.sleep(3)

for i in xrange(1000):
    print ''.join([str(random.randint(0,10)) for j in xrange(random.randint(3,20))])
    time.sleep(random.random() / 5)
    sys.stdout.flush()

