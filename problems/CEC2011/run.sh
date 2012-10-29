#!/bin/bash

MATLAB_PATH=/opt/MatlabR2011a/
export LD_LIBRARY_PATH=$MATLAB_PATH/bin/glnx86:$MATLAB_PATH/runtime/glnx86:`pwd`

env LD_LIBRARY_PATH=$MATLAB_PATH/bin/glnxa64/:$LD_LIBRARY_PATH ./test_function
