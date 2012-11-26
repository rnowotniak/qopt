#!/bin/bash

rm -fr /tmp/iQIEA-accordance/*
mkdir -p /tmp/iQIEA-accordance

for i in `seq -f '%02g' 1 50`; do
	./rcqiea.py 3 .23 .785 2>&1 | tee /tmp/iQIEA-accordance/$i.log
done

