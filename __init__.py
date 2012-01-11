
import time
import os

def tic():
    global timer_start
    timer_start = time.time()

def toc():
    return time.time() - timer_start

path = os.path.dirname(os.path.realpath(__file__))

