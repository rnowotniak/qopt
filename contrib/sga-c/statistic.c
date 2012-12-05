/*----------------------------------------------------------------------------*/
/* statistic.c - compute the fitness statistics                               */
/*----------------------------------------------------------------------------*/

#include "external.h"

statistics(pop)
/* Calculate population statistics */
struct individual *pop;
{
    int i, j;
    
    sumfitness = 0.0;
    min = pop[0].fitness;
    max = pop[0].fitness;

    /* Loop for max, min, sumfitness */
    for(j = 0; j < popsize; j++)
    {
        sumfitness = sumfitness + pop[j].fitness;               /* Accumulate */
        if(pop[j].fitness > max) max = pop[j].fitness;         /* New maximum */
        if(pop[j].fitness < min) min = pop[j].fitness;         /* New minimum */

        /* new global best-fit individual */
        if(pop[j].fitness > bestfit.fitness) 
	  {
	    for(i = 0; i < chromsize; i++)
	      bestfit.chrom[i]      = pop[j].chrom[i];

            bestfit.fitness    = pop[j].fitness;
            bestfit.generation = gen;
	  }
      }

    /* Calculate average */
    avg = sumfitness/popsize;

    /* get application dependent stats */
    app_stats(pop);
}
