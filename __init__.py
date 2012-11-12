
import time
import os

def tic():
    global timer_start
    timer_start = time.time()

def toc():
    return time.time() - timer_start

path = os.path.dirname(os.path.realpath(__file__))

def int2bin(n, count=24):
    return "".join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

