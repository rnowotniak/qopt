
# http://www.mail-archive.com/cython-dev@codespeak.net/msg07189.html

import ctypes
import os
import qopt
import numpy

from qopt.framework cimport ProblemDouble, ProblemCpp

cdef extern from "test_func.h":
    void test_func(double *x, double *f, int nx, int mx,int func_num)

cdef extern from "cec2013.h":
    cdef cppclass CEC2013cpp "CEC2013" (ProblemCpp[double,double]):
        CEC2013cpp(int fnum)

DEF MAXDIM = 10

fname = 'problems/CEC2013/cec13ccode/input_data/shift_data.txt'

cdef double evaluate_cec2013(int fnum, x):
    cdef double res[1]
    cdef double arg[MAXDIM]
    assert len(x) <= MAXDIM
    for i in xrange(len(x)):
        arg[i] = x[i]
    test_func(arg, res, len(x), 1, fnum)
    return res[0]

cdef class CEC2013(ProblemDouble):
    cdef long double (*r_evaluator) (long double *x,int n)
    cdef int fnum
    optimum = [float(x) for x in open(qopt.path(fname),'r').readlines()[0].split()][:MAXDIM]
    def __cinit__(self, int fnum):
        self.fnum = fnum
        self.thisptr = new CEC2013cpp(fnum)
    def evaluate(self, x):
        if type(x[0]) == numpy.ndarray:
            assert len(x[0].shape) == 2 and x[0].shape == x[1].shape
            oldpath = os.getcwd()
            os.chdir(qopt.path('problems/CEC2013/cec13ccode'))
            res = numpy.zeros(x[0].shape)
            for i in xrange(x[0].shape[0]):
                for j in xrange(x[0].shape[1]):
                    res[i][j] = evaluate_cec2013(self.fnum, (x[0][i][j], x[1][i][j]))
            os.chdir(oldpath)
            return res
        oldpath = os.getcwd()
        os.chdir(qopt.path('problems/CEC2013/cec13ccode'))
        res = evaluate_cec2013(self.fnum, x)
        os.chdir(oldpath)
        return res
