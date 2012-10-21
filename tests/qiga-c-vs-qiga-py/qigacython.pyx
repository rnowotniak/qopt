import time

import qopt.framework as qopt

cdef extern from "qiga.h":
    void qiga()
    float bestval
    void initialize()
    void observe()
    void repair()
    void evaluate()
    void update()
    void storebest()
    cdef cppclass QIGA:
        int popsize
        float Q[10][250]
        float getQ0()

def run():
    print 'qiga'

    start_tm = time.time()

    REPEAT = 100

    for rep in xrange(REPEAT):
        qiga()
        print bestval

    stop_tm = time.time()

    print '%g seconds\n' % (stop_tm - start_tm)


cdef class QIGAcpp:
    cdef QIGA *thisptr
    def __cinit__(self):
        self.thisptr = new QIGA()
    def getQ0(self):
        return self.thisptr.Q[0][0]
    property Q0:
        def __get__(self): return self.thisptr.Q[0][0]


class QIGAa(qopt.EA):

    def initialize(self):
        global bestval
        bestval = -1
        initialize()
        observe()
        repair()
        evaluate()
        storebest()

    def generation(self):
        observe()
        repair()
        evaluate()
        update()
        storebest()
        #self.bestval = bestval

    @property
    def best(self):
        return bestval

def start():
    q = QIGAa()
    q.tmax = 500

    start_tm = time.time()
    REPEAT = 100
    for rep in xrange(REPEAT):
        q.run()
        print q.bestval

    stop_tm = time.time()

    print '%g seconds\n' % (stop_tm - start_tm)



