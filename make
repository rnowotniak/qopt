#!/bin/bash

function check() {
	if [ "$?" -eq 0 ]; then
		echo OK
	else
		echo Compilation FAILED
		exit 1
	fi
}

cd algorithms
cython --cplus _algorithms.pyx
check
g++ -shared -o _algorithms.so _algorithms.cpp C/qiga.cpp `python-config --cflags`
check
cd ..

cd problems
cython --cplus _knapsack.pyx
check
g++ -shared -o _knapsack.so knapsack_.cpp _knapsack.cpp `python-config --cflags`
check
cd CEC2005
bash genlibs.sh
cd ..
cd ..


echo All well done!

