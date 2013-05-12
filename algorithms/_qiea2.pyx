#
# Copyright (C) 2012   Robert Nowotniak
#


import time
import qopt
import numpy as np
cimport numpy as cnp

cimport libc.stdlib
cimport libc.string

libc.stdlib.srand(time.time())

cnp.import_array()

from qopt.framework cimport ProblemDouble, ProblemCpp

cdef extern from "qiea2.h":
    cdef cppclass QIEA2cpp "QIEA2":
        QIEA2cpp(int chromlen, int popsize, int K)
        int t
        int chromlen
        int popsize
        int K
        float delta
        float XI
        #int tmax
        double bestval
        double *Q
        char **P
        double *fvals
        double *best
        ProblemCpp[double,double] *problem
        void bqigao()
        void initialize()
        void observe()
        void evaluate()
        void update()
        void storebest()


cdef class __QIEA2cpp:
    cdef QIEA2cpp *thisptr

    def __cinit__(self, int chromlen, int popsize = 10, int K = -1):
        print "__QIEA2cpp constructor"
        if K == -1:
            K = popsize
        self.thisptr = new QIEA2cpp(chromlen, popsize, K)
    def __dealloc__(self):
        del self.thisptr

    def _initialize(self):
        self.thisptr.initialize()
    def _observe(self):
        self.thisptr.observe()
    def _update(self):
        self.thisptr.update()
    def _storebest(self):
        self.thisptr.storebest()
    def _evaluate(self):
        self.thisptr.evaluate()

    # property mi:
    #     def __get__(self): return self.thisptr.mi
    #     def __set__(self, float mi): self.thisptr.mi = mi

    property delta:
        def __get__(self): return self.thisptr.delta
        def __set__(self, delta): self.thisptr.delta = delta
    property XI:
        def __get__(self): return self.thisptr.XI
        def __set__(self, XI): self.thisptr.XI = XI
    property t:
        def __get__(self): return self.thisptr.t
        def __set__(self, t): self.thisptr.t = t
    property popsize:
        def __get__(self): return self.thisptr.popsize
    property K:
        def __get__(self): return self.thisptr.K
    # property tmax:
    #     def __get__(self): return self.thisptr.tmax
    #     def __set__(self, int tmax): self.thisptr.tmax = tmax
    property best:
        def __get__(self):
            result = []
            for i in xrange(self.thisptr.chromlen):
                result.append(self.thisptr.best[i])
            return result
        def __set__(self, char *val): libc.string.memcpy(self.thisptr.best, val, len(val))
    property bestval:
        def __get__(self): return self.thisptr.bestval
        def __set__(self, val): self.thisptr.bestval = val
    property fvals:
        def __get__(self):
            cdef cnp.npy_intp shape[1]
            shape[0] = <cnp.npy_intp> self.thisptr.popsize
            ndarray = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_DOUBLE, self.thisptr.fvals)
            return ndarray
    property problem:
        def __set__(self, ProblemDouble p):
            self.thisptr.problem = p.thisptr
    property Q:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = <cnp.npy_intp> self.thisptr.popsize
            shape[1] = <cnp.npy_intp> self.thisptr.chromlen/2 * 5
            ndarray = cnp.PyArray_SimpleNewFromData(2, shape, cnp.NPY_DOUBLE, self.thisptr.Q)
            # ndarray.base = ... ???
            # Py_INCREF(self) ... ???
            return ndarray
    property P:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = <cnp.npy_intp> self.thisptr.popsize
            shape[1] = <cnp.npy_intp> self.thisptr.chromlen
            ndarray = cnp.PyArray_SimpleNewFromData(2, shape, cnp.NPY_DOUBLE, self.thisptr.P)
            return ndarray


class QIEA2(__QIEA2cpp, qopt.EA):

    def __init__(self, int chromlen, int popsize = 10, int K = -1):
        qopt.EA.__init__(self)

    def initialize(self):
        self.bestval = -1
        self._initialize()
        self._observe()
        self._evaluate()
        self._storebest()

    def generation(self):
        #print 'gen %d' % self.t
        self._observe()
        #self._evaluate()  # performed in _update  # tylko to zakomentowane
        self._update()
        self._storebest()


