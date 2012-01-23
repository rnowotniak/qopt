#!/bin/bash
#
# Temporary files: /tmp/sga/*
# Output files: /tmp/plot-f*.pdf
#

FUNCTIONS=25

echo "Running SGA in parallel..."
for FNUM in `seq $FUNCTIONS`; do
    echo "f$FNUM"
    ./sga.sh $FNUM &
done

wait

echo "Running iQIEA in parallel..."
for FNUM in `seq $FUNCTIONS`; do
    echo "f$FNUM"
    python run-iqiea-and-plot.py $FNUM > /dev/null &
done

wait

echo All done

