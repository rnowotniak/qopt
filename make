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

#INCLUDE="-I../C -fPIC"
INCLUDE="-I../C -fPIC -D QOPT_PATH=\"$PWD/\"  -I/usr/local/lib/python2.6/dist-packages/numpy/core/include"
export DEBUG=0

# framework
cython --cplus framework.pyx
check
g++  `python-config --cflags` -I C/ -fPIC -D QOPT="\"$PWD/\"" -I/usr/local/lib/python2.6/dist-packages/numpy/core/include -shared framework.cpp -o framework.so
check

# algorithms
cd algorithms
cython --cplus _myrqiea2.pyx
check
g++  `python-config --cflags` $INCLUDE -shared -o _myrqiea2.so _myrqiea2.cpp ../C/myrqiea2.cpp
check

cython --cplus _qiea1.pyx
check
g++  `python-config --cflags` $INCLUDE -shared -o _qiea1.so _qiea1.cpp ../C/qiea1.cpp
check

cython --cplus _qiea2.pyx
check
g++ `python-config --cflags`  $INCLUDE -shared -o _qiea2.so _qiea2.cpp ../C/qiea2.cpp
check

cython --cplus _algorithms.pyx
check
g++ `python-config --cflags`  $INCLUDE -shared -o _algorithms.so _algorithms.cpp ../C/qiga.cpp ../C/bqigao.cpp ../C/bqigao2.cpp
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
g++ `python-config --cflags`  $INCLUDE -shared -o _knapsack.so ../C/knapsack.cpp _knapsack.cpp
check
cython --cplus _sat.pyx
check
g++ `python-config --cflags`  $INCLUDE -shared -o _sat.so ../C/sat.cpp _sat.cpp
check
cython --cplus _func1d.pyx
check
g++  `python-config --cflags` $INCLUDE -shared -o _func1d.so _func1d.cpp ../C/functions1d.cpp -I../contrib/alglib/src ../contrib/alglib/src/*.o -lgmp
check

# numerical
cd CEC2005
bash genlibs.sh
check
cd ..
cython --cplus _cec2013.pyx
check
g++  `python-config --cflags` $INCLUDE -I CEC2013/cec13ccode -shared -o _cec2013.so _cec2013.cpp CEC2013/cec13ccode/test_func.cpp
check
cython --cplus _cec2005.pyx
check
g++ `python-config --cflags`  $INCLUDE -shared -o _cec2005.so _cec2005.cpp
check
cython --cplus _cec2011.pyx
check
g++  `python-config --cflags` $INCLUDE -I CEC2011 -shared -o _cec2011.so _cec2011.cpp  CEC2011/mCEC_Function.o -lCEC2011 -L./CEC2011
check

cd ..


echo All well done!

