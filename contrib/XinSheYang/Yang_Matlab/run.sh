#!/bin/bash

FUNCTIONS=25
REP=25

echo "Running BAT in parallel..."
for FNUM in `seq 15 $FUNCTIONS`; do # XXX
	echo "f$FNUM"
	mkdir -p "/var/tmp/bat-f${FNUM}"
	R=0
	while [ $R -lt $REP ]; do
		echo "tic;bat_algorithm(${FNUM});toc;" | \
			/opt/MatlabR2011a/bin/matlab -nodesktop -nojvm > "/var/tmp/bat-f${FNUM}/run${R}.txt" &
		#case $R in
			#8 | 16 )
				#wait
				#;;
		#esac
		R=$[R+1]
	done
	wait
done

echo All done


