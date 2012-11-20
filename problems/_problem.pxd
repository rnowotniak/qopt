
cdef extern from "../framework.h":
    cdef cppclass ProblemCpp "Problem":
        float evaluator (char*, int)
        void repairer (char*, int)
        long double r_evaluator(long double *, int)

cdef class Problem:
    cdef ProblemCpp *thisptr

