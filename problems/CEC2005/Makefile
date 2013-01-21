# Makefile for compiling test functions
# Tested on the following system:
# Compiler name and version: gcc-3.3.4
# Operating system name and version: Gentoo Linux 2004.3
# Kernel name and version: 2.6.9-gentoo-r9
# Machine architecture: i686
# Last edited on: 10 March 2005
# Author: Santosh Tiwari (tiwaris@iitk.ac.in)

CC=/usr/bin/gcc

LD=/usr/bin/gcc

RM=/bin/rm -f

CFLAGS=-Df$(FNUM) -DDEBUG=$(DEBUG) -Wall -Werror -ansi -pedantic -g

#CFLAGS=-O2 -march=pentium4 -pipe -fomit-frame-pointer
#LDFLAGS=-s

# OBJS := $(patsubst %.c,%.o,$(wildcard *.c))
OBJS := aux.o def1.o def2.o def3.o def4.o main.o rand.o

MAIN = main

all: $(MAIN)

$(MAIN): $(OBJS)
	$(LD) $(LDFLAGS) $(OBJS) -o $(MAIN) -lm -L./sprng/lib -llcg

%.o: %.c global.h sub.h rand.h
	$(CC) $(CFLAGS) -c $<

clean:
	$(RM) $(OBJS) core.* *~ *.out *.o

