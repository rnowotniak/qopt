
cdef extern from "framework.h":
    cdef cppclass ProblemCpp "Problem" [ARGTYPE,RESTYPE]:
        RESTYPE evaluator (ARGTYPE*, int)
        void repairer (ARGTYPE*, int)

cdef class Problem:
    cdef ProblemCpp[char,float] *thisptr

cdef class ProblemDouble:
    cdef ProblemCpp[double,double] *thisptr

