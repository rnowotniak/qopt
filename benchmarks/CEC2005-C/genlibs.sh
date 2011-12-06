#!/bin/bash

rm -f libf*.so

for FNUM in `seq 25`; do
	export FNUM
	make clean
	make
	gcc -shared -L/usr/lib/python2.6/config -lpthread -ldl -lutil -lm -lpython2.6 \
		-I/usr/include/python2.6 -I/usr/include/python2.6 \
		aux.o def1.o def2.o def3.o def4.o main.o rand.o \
		-o libf${FNUM}.so \
		-lm -L./sprng/lib -llcg
done

strip libf*.so

