
from qopt.problems._problem cimport Problem, ProblemCpp


cdef extern from "C/sat.h":
    cdef cppclass SATcpp "SAT" (ProblemCpp):
        SATcpp(char *)

cdef class SatProblem(Problem):
    def __cinit__(self, fname):
        self.thisptr = new SATcpp(fname);

    def evaluate(self, char *k):
        return self.thisptr.evaluator(k, len(k))

