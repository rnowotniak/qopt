
from qopt.problems._problem cimport Problem, ProblemCpp


cdef extern from "C/sat.h":
    cdef cppclass SATcpp "SAT" (ProblemCpp):
        int numatom
        SATcpp(char *) except +SyntaxError

cdef class SatProblem(Problem):
    def __cinit__(self, fname):
        self.thisptr = new SATcpp(fname);

    def evaluate(self, k):
        atoms = (<SATcpp *>self.thisptr).numatom
        if len(k) != atoms:
            raise Exception('Incompatibile string length (%d != %d)' % \
                    (len(k), atoms))
        return self.thisptr.evaluator(k, len(k))

    property numatom:
        def __get__(self): return (<SATcpp *>self.thisptr).numatom

