
from qopt.problems._problem cimport Problem, ProblemCpp

cdef extern from "C/knapsack.h":
    cdef cppclass KnapsackProblemCpp "KnapsackProblem" (ProblemCpp[char,float]):
        pass

cdef class KnapsackProblem(Problem):
    def __cinit__(self):
        self.thisptr = new KnapsackProblemCpp()

    def evaluate(self, k):
        return self.thisptr.evaluator(k, len(k))

