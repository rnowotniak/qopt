
from qopt.algorithms._algorithms cimport Problem, evaluator_t, repairer_t
# from Problem cimport *

# ctypedef float (*evaluator_t) (char*,int)
# ctypedef void (*repairer_t) (char*,int)
# cdef class Problem:
#     cdef evaluator_t evaluator
#     cdef repairer_t repairer

cdef extern from "C/sat.h":
    cdef cppclass SATcpp "SAT":
        SATcpp(char *)
        float evaluate(char *, int)

    #void c_repairKnapsack "repairKnapsack" (char *x, int)
    #float c_fknapsack "fknapsack" (char *, int)

cdef class SatProblem(Problem):
    cdef SATcpp *thisptr
    def __cinit__(self, fname):
        self.thisptr = new SATcpp(fname);
        #self.evaluator = self.thisptr.evaluate
        #self.repairer = c_repairKnapsack

    def evaluate(self, char *k):
        return self.thisptr.evaluate(k, len(k))
        #return self.evaluator(k, len(k))

