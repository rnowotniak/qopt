#!/bin/bash

export PATH="$PATH:/opt/MatlabR2011a/bin"

MATLAB_PATH=/opt/MatlabR2011a/

export LD_LIBRARY_PATH=$MATLAB_PATH/bin/glnx86:$MATLAB_PATH/runtime/glnx86


mcc -W lib:CEC2011 -T link:lib matlab/*.m -d .
mv CEC2011.so libCEC2011.so
g++ -c mCEC_Function.cpp -I $MATLAB_PATH/extern/include/
g++ mCEC_Function.o test_function.cpp -o test_function -lCEC2011 -L. -L$MATLAB_PATH/runtime/glnx86 -L$MATLAB_PATH/bin/glnx86/ -lmwmclmcrrt

g++ -c mCEC_Function.o test_function.cpp -o test_function.o -lCEC2011 -L. -L$MATLAB_PATH/runtime/glnx86 -L$MATLAB_PATH/bin/glnx86/ -lmwmclmcrrt

