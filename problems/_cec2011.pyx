
# http://www.mail-archive.com/cython-dev@codespeak.net/msg07189.html

import ctypes
import os

from qopt.problems._problem cimport Problem, ProblemCpp

cdef extern from "mCEC_Function.h":
    ctypedef double FIELD_TYPE
    int Initial_CEC2011_Cost_Function()
    void Terminate_CEC2011_Cost_Function()
    void cost_function15(FIELD_TYPE *x, FIELD_TYPE *f)

cdef class CEC2011(Problem):

    cdef long double (*r_evaluator) (long double *x,int n)

    def __cinit__(self, int fnum):
        Initial_CEC2011_Cost_Function()
        # TODO:  set fnum and r_evaluator accordingly

    def evaluate(self, x):
        cdef FIELD_TYPE tab[1000]
        os.chdir('problems/CEC2011')
        cdef int i
        for i in xrange(len(x)):
            tab[i] = x[i]
        cdef FIELD_TYPE res[1]
        cost_function15(tab, res)
        os.chdir('../..')
        return res[0]

