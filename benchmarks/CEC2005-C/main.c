/* Sample main to demonstrate the use of various functions */
/* Please go through this file carefully */
/* It demonstrates the use of various routines */

# include <stdio.h>
# include <stdlib.h>

# include "global.h"
# include "sub.h"
# include "rand.h"

int main (int argc, char**argv)
{
	int i;
    long double *x;
    long double f;
    if (argc<3)
    {
        fprintf(stderr,"\n Usage ./main nfunc nreal\n");
        exit(0);
    }

	/* Assign nfunc and nreal in the begining */
    nfunc = (int)atoi(argv[1]);
    nreal = (int)atoi(argv[2]);

	if (nfunc<1)
    {
        fprintf(stderr,"\n Wrong value of 'nfunc' entered\n");
        exit(0);
    }
    if (nreal!=2 && nreal!=10 && nreal!=30 && nreal!=50)
    {
        fprintf(stderr,"\n Wrong value of 'nreal' entered, only 2, 10, 30, 50 variables are supported\n");
        exit(0);
    }
    printf("\n Number of basic functions = %d",nfunc);
    printf("\n Number of real variables  = %d",nreal);

	/* Call these routines to initialize random number generator */
	/* require for computing noise in some test problems */
	randomize();
    initrandomnormaldeviate();

	/* nreal and nfunc need to be initialized before calling these routines */
	/* Routine to allocate memory to global variables */
    allocate_memory();

	/* Routine the initalize global variables */
    initialize();

	/* For test problems 15 to 25, we need to calculate a normalizing quantity */
    /* The line (54) below should be uncommented only for functions 15 to 25 */
	/*calc_benchmark_norm();*/    /* Comment this line for functions 1 to 14 */

	/* Variable vector */
	x = (long double *)malloc(nreal*sizeof(long double));

	for (i=0; i<nreal; i++)
	{
		printf("\n Enter the value of variable x[%d] : ",i+1);
		scanf("%Lf",&x[i]);
	}
	f = calc_benchmark_func(x);
	printf("\n Objective value = %1.15LE",f);

	/* Routine to free the memory allocated at run time */
	free_memory();

	free (x);
    printf("\nRoutine exited without any error.\n");
    return(1);
}
