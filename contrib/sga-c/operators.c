/*----------------------------------------------------------------------------*/
/* operators.c - Genetic Operators: Crossover and Mutation                    */
/*----------------------------------------------------------------------------*/

#include "external.h"

mutation(child)
unsigned *child;
/* Mutate an allele w/ pmutation, count # of mutations */
{
    int j, k, stop;
    unsigned mask, temp = 1;

    for(k = 0; k < chromsize; k++)
    {
        mask = 0;
        if(k == (chromsize-1))
            stop = lchrom - (k*UINTSIZE);
        else
            stop = UINTSIZE;
        for(j = 0; j < stop; j++)
        {
            if(flip(pmutation))
            {
                mask = mask|(temp<<j);
                nmutation++;
            }
        }
        child[k] = child[k]^mask;
    }
}


int crossover (parent1, parent2, child1, child2)
unsigned *parent1, *parent2, *child1, *child2;
/* Cross 2 parent strings, place in 2 child strings */
{
    int j, jcross, k;
    unsigned mask, temp;

    /* Do crossover with probability pcross */
    if(flip(pcross))
    {
        jcross = rnd(1 ,(lchrom - 1));/* Cross between 1 and l-1 */
        ncross++;
        for(k = 1; k <= chromsize; k++)
        {
            if(jcross >= (k*UINTSIZE))
            {
                child1[k-1] = parent1[k-1];
                child2[k-1] = parent2[k-1];
            }
            else if((jcross < (k*UINTSIZE)) && (jcross > ((k-1)*UINTSIZE)))
            {
                mask = 1;
                for(j = 1; j <= (jcross-1-((k-1)*UINTSIZE)); j++)
                {
                    temp = 1;
                    mask = mask<<1;
                    mask = mask|temp;
                }
                child1[k-1] = (parent1[k-1]&mask)|(parent2[k-1]&(~mask));
                child2[k-1] = (parent1[k-1]&(~mask))|(parent2[k-1]&mask);
            }
            else
            {
                child1[k-1] = parent2[k-1];
                child2[k-1] = parent1[k-1];
            }
        }
    }
    else
    {
        for(k = 0; k < chromsize; k++)
        {
            child1[k] = parent1[k];
            child2[k] = parent2[k];
        }
        jcross = 0;
    }
    return(jcross);
}
