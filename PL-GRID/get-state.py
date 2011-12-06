#!/usr/bin/python

import sys
import tempfile
import os
import subprocess

# tmp = tempfile.mkstemp()
# print tmp
# sys.stdout.flush()

p = subprocess.Popen("ssh plgrid 'pbsnodes 2>&1'", shell=True, stdout=subprocess.PIPE)
data = p.communicate()[0]
print data

# f = open(tmp[1], 'w')
# print data
# sys.stdout.flush()
# f.write(data)
# f.close()
# 
# os.unlink(tmp[1])

