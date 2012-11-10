#
# Copyright (C) 2012   Robert Nowotniak
#


import time
import qopt.framework as qopt
import numpy as np
cimport numpy as cnp

cimport libc.stdlib

libc.stdlib.srand(time.time())

cnp.import_array()

# ctypedef float (*evaluator_t) (char*,int)
# ctypedef void (*repairer_t) (char*,int)

# cdef class Problem:
#     cdef evaluator_t evaluator
#     cdef repairer_t repairer

cdef extern from "C/qiga.h":
    ctypedef float (*evaluator_t) (char*,int)
    ctypedef void (*repairer_t) (char*,int)
    cdef cppclass QIGAcpp "QIGA":
        int popsize
        int chromlen
        float bestval
        float *Q
        char **P
        float *fvals
        char *best
        float lookup_table[2][2][2]
        float signs_table[2][2][2][4]
        evaluator_t evaluator
        repairer_t repairer
        void qiga()
        void initialize()
        void observe()
        void repair()
        void evaluate()
        void update()
        void storebest()


class EA:
    def __init__(self):
        print 'EA __init__'
        self.t = 0

    def run(self):
        self.t = 0
        self.initialize()
        while self.t < self.tmax:
            self.t += 1
            self.generation()


cdef class __QIGAcpp:
    cdef QIGAcpp *thisptr

    def __cinit__(self):
        self.thisptr = new QIGAcpp()
    def __dealloc__(self):
        del self.thisptr

    def _qiga(self):
        self.thisptr.qiga()
    def _initialize(self):
        self.thisptr.initialize()
    def _observe(self):
        self.thisptr.observe()
    def _repair(self):
        self.thisptr.repair()
    def _update(self):
        self.thisptr.update()
    def _storebest(self):
        self.thisptr.storebest()
    def _evaluate(self):
        self.thisptr.evaluate()

    property popsize:
        def __get__(self): return self.thisptr.popsize
    property bestval:
        def __get__(self): return self.thisptr.bestval
        def __set__(self, val): self.thisptr.bestval = val
    property fvals:
        def __get__(self):
            cdef cnp.npy_intp shape[1]
            shape[0] = <cnp.npy_intp> self.thisptr.popsize
            ndarray = cnp.PyArray_SimpleNewFromData(1, shape, cnp.NPY_FLOAT, self.thisptr.fvals)
            return ndarray
    property problem:
        def __set__(self, Problem e):
            self.thisptr.evaluator = e.evaluator
            self.thisptr.repairer = e.repairer
    property lookup_table:
        def __get__(self):
            cdef cnp.npy_intp shape[3]
            shape[0] = <cnp.npy_intp> 2
            shape[1] = <cnp.npy_intp> 2
            shape[2] = <cnp.npy_intp> 2
            ndarray = cnp.PyArray_SimpleNewFromData(3, shape, cnp.NPY_FLOAT, self.thisptr.lookup_table)
            return ndarray
    property Q:
        def __get__(self):
            cdef cnp.npy_intp shape[2]
            shape[0] = <cnp.npy_intp> self.thisptr.popsize
            shape[1] = <cnp.npy_intp> self.thisptr.chromlen
            ndarray = cnp.PyArray_SimpleNewFromData(2, shape, cnp.NPY_FLOAT, self.thisptr.Q)
            # ndarray.base = ... ???
            # Py_INCREF(self) ... ???
            return ndarray
    property P:
        def __get__(self):
            res = []
            for i in xrange(self.thisptr.popsize):
                res.append(self.thisptr.P[i][:self.thisptr.chromlen])
            return res


class QIGA(__QIGAcpp, EA):

    def __init__(self):
        EA.__init__(self)

    def initialize(self):
        self.bestval = -1
        self._initialize()
        self._observe()
        self._repair()
        self._evaluate()
        self._storebest()

    def generation(self):
        self._observe()
        self._repair()
        self._evaluate()
        self._update()
        self._storebest()

