
ctypedef float (*evaluator_t) (char*,int)
ctypedef void (*repairer_t) (char*,int)

cdef class Problem:
    cdef evaluator_t evaluator
    cdef repairer_t repairer

