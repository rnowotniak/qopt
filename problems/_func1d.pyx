
from qopt.problems._problem cimport Problem, ProblemCpp

cdef extern from "functions1d.h":
    double func1(double x)
    double func2(double x)
    double func3(double x)
    float func1_b(char *,int)
    float func2_b(char *,int)
    float func3_b(char *,int)
    float getx(char *s, int len, float mi, float ma)
    cdef cppclass Functions1DProblemCpp "Functions1DProblem" (ProblemCpp[char,float]):
        Functions1DProblemCpp(int fnum)


cdef class Func1D(Problem):
    cdef double (*f)(double)
    cdef float mi
    cdef float ma
    def __cinit__(self, int fnum):
        self.thisptr = new Functions1DProblemCpp(fnum)
        if fnum == 1:
            self.f = func1
            # self.evaluator = func1_b
            self.mi = 0
            self.ma = 200
        elif fnum == 2:
            self.f = func2
            # self.evaluator = func2_b
            self.mi = -5
            self.ma = 5
        elif fnum == 3:
            self.f = func3
            # self.evaluator = func3_b
            self.mi = 0
            self.ma = 17
    def evaluate(self, double x):
        return self.f(x)
    def getx(self, s):
        return getx(s, len(s), self.mi, self.ma)

