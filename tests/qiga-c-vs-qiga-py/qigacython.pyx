#
# Copyright (C) 2012   Robert Nowotniak
#


import time
import qopt.framework as qopt
import numpy as np
cimport numpy as cnp
#from cpython cimport Py_INCREF, PyObject

cimport libc.stdlib

libc.stdlib.srand(time.time())

cnp.import_array()

cdef extern from "knapsack.h":
    void c_repairKnapsack "repairKnapsack" (char *x, int)
    float c_fknapsack "fknapsack" (char *, int)

cdef extern from "mCEC_Function.h":
    ctypedef double FIELD_TYPE
    int Initial_CEC2011_Cost_Function()
    void Terminate_CEC2011_Cost_Function()
    void cost_function1(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function2(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function3(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function4(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function5(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function6(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function7(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function8(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function9(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function10(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function11_5(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function11_10(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function12_6(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function12_13(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function12_15(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function12_40(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function12_140(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function13_1(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function13_2(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function13_3(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function14(FIELD_TYPE *x, FIELD_TYPE *f)
    void cost_function15(FIELD_TYPE *x, FIELD_TYPE *f)

def cec2011_1(arg):
    Initial_CEC2011_Cost_Function()
    cdef FIELD_TYPE x[1000]
    cdef int i
    for i in xrange(len(arg)):
        x[i] = arg[i]
    cdef FIELD_TYPE res[1]
    for i in xrange(1000):
        cost_function15(x, res)
    return res[0]



cdef extern from "qiga.h":
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



cdef class Problem:
    cdef evaluator_t evaluator
    cdef repairer_t repairer

cdef class KnapsackProblem(Problem):
    def __cinit__(self):
        self.evaluator = c_fknapsack
        self.repairer = c_repairKnapsack

cdef float onemax(char *s, int l):
    cdef float res
    res = s[:l].count('1')
    return res

cdef class OneMaxProblem(Problem):
    def __cinit__(self):
        self.evaluator = onemax



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


class bQIGAo(__QIGAcpp, qopt.EA):
    pass


def runcpp():
    print 'qiga.cpp'

    start_tm = time.time()

    REPEAT = 100

    q = __QIGAcpp()
    q.problem = KnapsackProblem()

    for rep in xrange(REPEAT):
        q._qiga()
        print q.bestval

    stop_tm = time.time()

    print '%g seconds\n' % (stop_tm - start_tm)


def testtime(alg):
    start_tm = time.time()
    REPEAT = 100
    for rep in xrange(REPEAT):
        alg.run()
        print alg.bestval
    stop_tm = time.time()
    print '%g seconds\n' % (stop_tm - start_tm)

