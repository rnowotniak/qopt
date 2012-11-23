FLAGS=-Wall -Winline -O3 -funroll-loops
#FLAGS=-g -Wall -pg
#FLAGS=-pg
#CC=gcc

all:	spy verify walksat simann 
spy: spy.c spy.h formula.h bk.h formula.o random.o bk.o
	${CC} ${FLAGS} spy.c random.o formula.o bk.o -o spy -lm
formula.o: formula.h formula.c
	${CC} ${FLAGS} -c formula.c -o formula.o
random.o: random.c random.h
	${CC} ${FLAGS} -c random.c -o random.o
bk.o: bk.c bk.h
	${CC} ${FLAGS} -c bk.c -o bk.o
simann: simann.c simann.h formula.h random.o
	${CC} ${FLAGS} simann.c random.o -o simann -lm
clean:
	rm -f core* *~ *.o  *.tmp* *.old*
binclean: clean
	rm -f walksat spy verify simann
walksat: walksat.c
	${CC} -lm -O6 walksat.c -o walksat
verify: verify.c
	${CC} -Wall verify.c -o verify
package: spy.c spy.h bk.c bk.h formula.h formula.c random.c simann.c simann.h walksat.c verify.c Makefile Copying Changelog README
	tar -cvzf SPY-1.2.tar.gz .
