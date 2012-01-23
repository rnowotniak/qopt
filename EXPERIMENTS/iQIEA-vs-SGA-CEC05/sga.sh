#!/bin/bash

FNUM=$1

mkdir -p /tmp/sga
mkdir -p /tmp/sga/f$FNUM

for r in `seq 5`; do
	echo $r
	python sga-pyevolve.py $FNUM | grep '^Gen\.' | sed 's!/! !g' | gawk '{print $2*100 " " $9}' \
		> /tmp/sga/f$FNUM/$r.txt
done

~/qopt/analysis/avg-files.py /tmp/sga/f$FNUM/*.txt > /tmp/sga/f$FNUM/avg

