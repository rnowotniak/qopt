#!/bin/bash

for A in `seq 1 9`; do
	rm -f /tmp/anim2.png
	./anim2.py $A lus-steps-1.log
	for n in `seq -f '%03g' $[A*10+1] $[(A+1)*10]`; do
		cp -fl /tmp/anim2.png /tmp/anim2-1-$n.png
	done
done

for A in `seq 1 9`; do
	rm -f /tmp/anim2.png
	./anim2.py $A lus-steps-2.log
	for n in `seq -f '%03g' $[A*10+1] $[(A+1)*10]`; do
		cp -fl /tmp/anim2.png /tmp/anim2-2-$n.png
	done
done

for A in `seq 1 9`; do
	rm -f /tmp/anim2.png
	./anim2.py $A lus-steps-3.log
	for n in `seq -f '%03g' $[A*10+1] $[(A+1)*10]`; do
		cp -fl /tmp/anim2.png /tmp/anim2-3-$n.png
	done
done


# mencoder "mf://anim2-1-*png" -mf fps=10 "mf://anim2-2-*png" -mf fps=10 "mf://anim2-3-*.png" -mf fps=10  -o movie.avi -ovc lavc -lavcopts vcodec=mjpeg
# ffmpeg -sameq -i movie.avi -f avi -vcodec wmv1   zz.avi


