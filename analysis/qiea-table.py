#!/usr/bin/python

import sys
import numpy as np
from math import log, ceil

Xsize = 2**5

print 'natural   r_size    r_dim     registers    mem (B)'
print '-' * 70

for reg_size in xrange(1,int(log(Xsize) / log(2) + 1)):
    dim = 2**reg_size
    registers = log(Xsize) / log(dim)
    natural = ('   ','(*)')[registers == int(registers)]
    registers = ceil(registers)
    mem = 4 * registers * dim
    mem = registers * dim
    print '%s        %4d   %7d        %2g        %7d' % (natural, reg_size, dim, registers, mem)


