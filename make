#!/bin/bash

if [ "$1" = clean ]; then
	rm -f framework.so framework.cpp
	rm -fr *.pyc
	exit 0
fi

function check() {
	if [ "$?" -eq 0 ]; then
		echo OK
	else
		echo Compilation FAILED
		exit 1
	fi
}

INCLUDE=-I../C

# framework
cython --cplus framework.pyx
check
g++ -I C/ -shared framework.cpp -o framework.so `python-config --cflags`
check

# algorithms
cd algorithms
cython --cplus _algorithms.pyx
check
g++ $INCLUDE -shared -o _algorithms.so _algorithms.cpp ../C/qiga.cpp ../C/bqigao.cpp ../C/bqigao2.cpp `python-config --cflags`
check
cd ..

# problems
cd problems

#cython --cplus _problem.pyx
#check
#g++ -shared -o _problem.so _problem.cpp `python-config --cflags`
#check

# combinatorial
cython --cplus _knapsack.pyx
check
g++ $INCLUDE -shared -o _knapsack.so ../C/knapsack.cpp _knapsack.cpp `python-config --cflags`
check
cython --cplus _sat.pyx
check
g++ $INCLUDE -shared -o _sat.so ../C/sat.cpp _sat.cpp `python-config --cflags`
check
cython --cplus _func1d.pyx
check
g++ $INCLUDE -shared -o _func1d.so _func1d.cpp ../C/functions1d.cpp -I../contrib/alglib/src  `python-config --cflags` ../contrib/alglib/src/*.o -lgmp
check

# numerical
cd CEC2005
#bash genlibs.sh
check
cd ..
cython --cplus _cec2005.pyx
check
g++ $INCLUDE -shared -o _cec2005.so _cec2005.cpp `python-config --cflags`
check
cython --cplus _cec2011.pyx
check
g++ $INCLUDE -I CEC2011 -shared -o _cec2011.so _cec2011.cpp  CEC2011/mCEC_Function.o  `python-config --cflags` -lCEC2011 -L./CEC2011
check

cd ..


echo All well done!

