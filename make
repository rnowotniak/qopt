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
g++ -shared -o _algorithms.so _algorithms.cpp qiga.cpp `python-config --cflags`
check
cd ..

cd problems
cython --cplus Problem.pyx
check
g++ -shared -o Problem.so Problem.cpp `python-config --cflags`
check
g++ -shared -o knapsack.so knapsack_.cpp knapsack.cpp `python-config --cflags`
check
cd ..


echo All well done!

