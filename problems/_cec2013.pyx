
# http://www.mail-archive.com/cython-dev@codespeak.net/msg07189.html

import ctypes
import os
import qopt

from qopt.framework cimport Problem, ProblemCpp

cdef extern from "test_func.h":
    void test_func(double *x, double *f, int nx, int mx,int func_num)


DEF MAXDIM = 10

fname = 'problems/CEC2013/cec13ccode/input_data/shift_data.txt'

cdef class CEC2013(Problem):
    cdef long double (*r_evaluator) (long double *x,int n)
    cdef int fnum
    optimum = [float(x) for x in open(qopt.path(fname),'r').readlines()[0].split()][:MAXDIM]
    def __cinit__(self, int fnum):
        self.fnum = fnum
    def evaluate(self, x):
        cdef double res[1]
        cdef double arg[MAXDIM]
        assert len(x) <= MAXDIM
        for i in xrange(len(x)):
            arg[i] = x[i]
        oldpath = os.getcwd()
        os.chdir(qopt.path('problems/CEC2013/cec13ccode'))
        test_func(arg, res, len(x), 1, self.fnum)
        # m++ -> possible speedup ?
        os.chdir(oldpath)
        return res[0]

