
from qopt.problems._problem cimport Problem, ProblemCpp

cdef extern from "C/knapsack.h":
    cdef cppclass KnapsackProblemCpp "KnapsackProblem" (ProblemCpp[char,float]):
        int items_count
        float capacity
        float (*items)[2]
        KnapsackProblemCpp(char *) except +SyntaxError

cdef class KnapsackProblem(Problem):
    def __cinit__(self, fname):
        self.thisptr = new KnapsackProblemCpp(fname)

    def evaluate(self, char *k):
        return self.thisptr.evaluator(k, len(k))

    def repair(self, char *k):
        self.thisptr.repairer(k, len(k))

    def evaluate2(self, char *k):
        cdef float weight = 0
        cdef float price = self.thisptr.evaluator(k, len(k))
        for i in xrange(len(k)):
            if k[i] == '1':
                weight += (<KnapsackProblemCpp*> self.thisptr).items[i][0]
        return (weight, price)

    property items_count:
        def __get__(self): return (<KnapsackProblemCpp*> self.thisptr).items_count

    property capacity:
        def __get__(self): return (<KnapsackProblemCpp*> self.thisptr).capacity
        def __set__(self, val): (<KnapsackProblemCpp*> self.thisptr).capacity = val

    def item(self, nr):
        cdef float *item = (<KnapsackProblemCpp*> self.thisptr).items[nr]
        return (item[0], item[1])

