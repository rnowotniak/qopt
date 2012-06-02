#!/bin/bash

FUNCTIONS=6
REP=25

echo "Running DE in parallel..."
for FNUM in `seq 15 24`; do
	echo "f$FNUM"
	mkdir -p "/var/tmp/de-f${FNUM}"
	R=0
	while [ $R -lt $REP ]; do
		echo "tic;de(${FNUM});toc;" | \
			/opt/MatlabR2011a/bin/matlab -nodesktop -nojvm > "/var/tmp/de-f${FNUM}/run${R}.txt" &
		R=$[R+1]
		echo "tic;de(${FNUM});toc;" | \
			/opt/MatlabR2011a/bin/matlab -nodesktop -nojvm > "/var/tmp/de-f${FNUM}/run${R}.txt" &
		R=$[R+1]
		wait
		#case $R in
			#8 | 16 )
				#wait
				#;;
		#esac
	done
	wait
done

echo All done


