#!/bin/bash

./search2 ../../data/func1d-15-space 15 | sort -k 2 -n -r | head -n 20 > func1d-15-bblocks &
./search2 ../../data/func1d-20-space 20 | sort -k 2 -n -r | head -n 20 > func1d-20-bblocks &
./search2 ../../data/func1d-25-space 25 | sort -k 2 -n -r | head -n 20 > func1d-25-bblocks &

./search2 ../../data/k15-space 15 | sort -k 2 -n -r | head -n 20 > k15-bblocks &
./search2 ../../data/k20-space 20 | sort -k 2 -n -r | head -n 20 > k20-bblocks &
./search2 ../../data/k25-space 25 | sort -k 2 -n -r | head -n 20 > k25-bblocks &

./search2 ../../data/sat15-space 15 | sort -k 2 -n -r | head -n 20 > sat15-bblocks &
./search2 ../../data/sat20-space 20 | sort -k 2 -n -r | head -n 20 > sat20-bblocks &
./search2 ../../data/sat25-space 25 | sort -k 2 -n -r | head -n 20 > sat25-bblocks &

wait

