
import time

def tic():
    global timer_start
    timer_start = time.time()

def toc():
    return time.time() - timer_start

