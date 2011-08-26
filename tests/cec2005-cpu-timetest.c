/*
 * Ten plik sluzy do zbadania szybkosci dzialania benchmarku CEC2005 w implementacji C na CPU
 */

# include <stdio.h>
# include <stdlib.h>

# include "global.h"
# include "sub.h"
# include "rand.h"

#include <sys/time.h>

#define REPEAT 512000

int main (int argc, char**argv)
{
	int i;
	long double *x;
	long double f;
	struct timeval tv1, tv2;
	struct timezone tz1, tz2;
	long int tdiff;

	nfunc = 1;
	nreal = 50;

	randomize();
	initrandomnormaldeviate();
	allocate_memory();
	initialize();

	x = (long double *)malloc(REPEAT*nreal*sizeof(long double));

	printf("\n\n");

	for (i=0; i<REPEAT * nreal; i++)
	{
		x[i] = rndreal(-100,100);
	}

	printf("\n\n");

	gettimeofday(&tv1, &tz1);
	for (i = 0; i < REPEAT; i++) {
		f = calc_benchmark_func(x + nreal * i);
		//printf("Objective value = %1Lf\n",f);
	}
	gettimeofday(&tv2, &tz2);
	tdiff = (tv2.tv_sec - tv1.tv_sec) * 1000000 + (tv2.tv_usec - tv1.tv_usec);
	printf("time: %ld microseconds\n", tdiff);

	free_memory();
	free (x);
	return(1);
}
