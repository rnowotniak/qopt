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

def run():
    print 'qiga'

    start_tm = time.time()

    REPEAT = 100

    for rep in xrange(REPEAT):
        qiga()
        print bestval

    stop_tm = time.time()

    print '%g seconds\n' % (stop_tm - start_tm)


class QIGA(qopt.EA):

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
        self.bestval = bestval

def start():
    q = QIGA()
    q.tmax = 500

    start_tm = time.time()
    REPEAT = 100
    for rep in xrange(REPEAT):
        q.run()
        print q.bestval

    stop_tm = time.time()

    print '%g seconds\n' % (stop_tm - start_tm)



