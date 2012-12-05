/*----------------------------------------------------------------------------*/
/* rselect.c:  roulette wheel selection.                                      */
/*----------------------------------------------------------------------------*/

#include "external.h"

select_memory()
{
}

select_free()
{
}

preselect()
{
    int j;

    sumfitness = 0;
    for(j = 0; j < popsize; j++) sumfitness += oldpop[j].fitness;
}


int select()
/* roulette-wheel selection */
{
    extern float randomperc();
    float sum, pick;
    int i;

    pick = randomperc();
    sum = 0;

    if(sumfitness != 0)
    {
        for(i = 0; (sum < pick) && (i < popsize); i++)
            sum += oldpop[i].fitness/sumfitness;
    }
    else
        i = rnd(1,popsize);

    return(i-1);
}
