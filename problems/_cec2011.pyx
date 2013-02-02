
# http://www.mail-archive.com/cython-dev@codespeak.net/msg07189.html

import ctypes
import os
import qopt

from qopt.framework cimport Problem, ProblemCpp

cdef extern from "mCEC_Function.h":
    ctypedef double FIELD_TYPE
    int Initial_CEC2011_Cost_Function()
    void Terminate_CEC2011_Cost_Function()
    void cost_function14(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function15(FIELD_TYPE *x, FIELD_TYPE *f)

cdef class CEC2011(Problem):

    cdef long double (*r_evaluator) (long double *x,int n)

    def __cinit__(self, int fnum):
        # XXX do something with fnum!
        assert fnum == 15
        Initial_CEC2011_Cost_Function()
        # TODO:  set fnum and r_evaluator accordingly

    def evaluate(self, x):
        cdef FIELD_TYPE tab[1000]
        path = os.getcwd()
        os.chdir(qopt.path('problems/CEC2011'))
        cdef int i
        for i in xrange(len(x)):
            tab[i] = x[i]
        cdef FIELD_TYPE res[1]
        cost_function15(tab, res) # XXX
        os.chdir(path)
        return res[0]

    def evaluateMessenger(self, x):
        cdef FIELD_TYPE tab[1000]
        path = os.getcwd()
        os.chdir(qopt.path('problems/CEC2011'))
        cdef int i
        for i in xrange(len(x)):
            tab[i] = x[i]
        cdef FIELD_TYPE res[1]
        cost_function14(tab, res)
        os.chdir(path)
        return res[0]

