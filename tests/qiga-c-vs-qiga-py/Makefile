

all:
	cython --cplus qigacython.pyx && \
	g++ -I ../../problems/CEC2011 -o qigacython.so qiga.cpp qigacython.cpp knapsack.cpp ../../problems/CEC2011/test_function.o  ../../problems/CEC2011/mCEC_Function.o   -shared `python-config --cflags` -lCEC2011 -L../../problems/CEC2011 && echo OK

