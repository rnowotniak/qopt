
from qopt.algorithms._algorithms cimport Problem, evaluator_t #, repairer_t

cdef extern from "C/functions1d.h":
    double func1(double x)
    double func2(double x)
    double func3(double x)

cdef class Func1D(Problem):
    def __cinit__(self):
        pass
    def evaluate(self, double x):
        return func1(x)


