/*----------------------------------------------------------------------------*/
/* app.c - application dependent routines, change these for different problem */
/*----------------------------------------------------------------------------*/

#include <math.h>
#include "external.h"

application()
/* this routine should contain any application-dependent computations */
/* that should be performed before each GA cycle. called by main()    */
{
}


app_data()
/* application dependent data input, called by init_data() */
/* ask your input questions here, and put output in global variables */
{
}


app_free()
/* application dependent free() calls, called by freeall() */
{
}


app_init()
/* application dependent initialization routine called by intialize() */
{
}


app_initreport()
/* Application-dependent initial report called by initialize() */
{
}


app_malloc()
/* application dependent malloc() calls, called by initmalloc() */
{
    char *malloc();
}


app_report()
/* Application-dependent report, called by report() */
{
}


app_stats(pop)
/* Application-dependent statistics calculations called by statistics() */
struct individual *pop;
{
}


objfunc(critter)
/* objective function used in Goldberg's book */
/* fitness function is f(x) = x**n, 
   normalized to range between 0 and 1,
   where x is the chromosome interpreted as an
   integer, and n = 10 */

struct individual *critter;
{
    unsigned mask=1;   /* mask for current bit */
    unsigned bitpos;   /* current bit position */
    unsigned tp;
    double pow(), bitpow, coef;
    int j, k, stop;
    int n = 10;

    critter->fitness = 0.0;
    coef = pow(2.0,(double) lchrom) - 1.0;
    coef = pow(coef, (double) n);

    /* loop thru number of bytes holding chromosome */
    for(k = 0; k < chromsize; k++)
    {
        if(k == (chromsize-1))
            stop = lchrom-(k*UINTSIZE);
        else
            stop = UINTSIZE;

        /* loop thru bits in current byte */
        tp = critter->chrom[k];
        for(j = 0; j < stop; j++)
        {
            bitpos = j + UINTSIZE*k; 
            /* test for current bit 0 or 1 */
            if((tp&mask) == 1)
            {
                bitpow = pow(2.0,(double) bitpos);
                critter->fitness = critter->fitness + bitpow;
            }
            tp = tp>>1;
        }
    }

    /* At this point, fitness = x */
    /* Now we must raise x to the n */
    critter->fitness = pow(critter->fitness,(double) n);

    /* normalize the fitness */
    critter->fitness = critter->fitness/coef;
}
