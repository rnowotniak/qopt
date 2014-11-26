#!/usr/bin/python

import sys
import re

while True:
    line=sys.stdin.readline()
    if not line:
        break
    result = re.sub(r'.*dim\d+-(.*)-(.*)\.npy.*', r'(\1,\2),', line)
    print result,

