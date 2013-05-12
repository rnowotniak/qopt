#!/usr/bin/python

import sys

from qopt.algorithms import QIEA2
from qopt.problems import CEC2013

K = 2
qiea2 = QIEA2(2, 10, 10)
qiea2.tmax = 100000 / K
fun = CEC2013(1)
qiea2.problem = fun

qiea2.delta = .9991
qiea2.XI = .2

#print fun.evaluate((
#    -90.028719906423945, -37.696144694077503, -83.881449882655559, 81.728825136564652, 
#      42.015512291015696, 69.905669103519841, -51.592264800711931, 31.986440145138928, 
#        -101.05137960062059, -22.303245202053802
#        ))
#

qiea2.run()

print qiea2.bestval

