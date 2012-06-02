#!/bin/bash
#

FUNCTIONS=25
REP=25

echo "Running SGA in parallel..."
for FNUM in `seq $FUNCTIONS`; do
	echo "f$FNUM"
	mkdir -p "/var/tmp/sga-f${FNUM}"
	for R in `seq $REP`; do
		time python experiment.py $FNUM > "/var/tmp/sga-f${FNUM}/run${R}.txt" &
	done
done

wait

# echo "Running iQIEA in parallel..."
# for FNUM in `seq $FUNCTIONS`; do
# 	echo "f$FNUM"
# 	python run-iqiea-and-plot.py $FNUM > /dev/null &
# done
# 
# wait

echo All done

