/*----------------------------------------------------------------------------*/
/* srselect.c - contains select_memory, select_free, 
/*              preselect, and select, stochastic remainder method            */
/*----------------------------------------------------------------------------*/

#include "external.h"

static int *choices, nremain;
static float *fraction;

select_memory()
{
  /* allocates auxiliary memory for stochastic remainder selection */

  unsigned nbytes;
  char     *malloc();
  int j;

  nbytes = popsize*sizeof(int);
  if((choices = (int *) malloc(nbytes)) == NULL)
    nomemory(stderr,"choices");
  nbytes = popsize*sizeof(float);
  if((fraction = (float *) malloc(nbytes)) == NULL)
    nomemory(stderr,"fraction");
}
  
select_free()
{
  /* frees auxiliary memory for stochastic remainder selection */
  free(choices);
  free(fraction);
}

preselect()
/* preselection for stochastic remainder method */
{
    int j, jassign, k;
    float expected;

    if(avg == 0)
    {
        for(j = 0; j < popsize; j++) choices[j] = j;
    }
    else
    {
        j = 0;
        k = 0;

        /* Assign whole numbers */
        do 
        {
            expected = ((oldpop[j].fitness)/avg);
            jassign = expected; 
            /* note that expected is automatically truncated */
            fraction[j] = expected - jassign;
            while(jassign > 0)
            {
                jassign--;
                choices[k] = j;
                k++;
            }
            j++;
        }
        while(j < popsize);
        
        j = 0;
        /* Assign fractional parts */
        while(k < popsize)
        { 
            if(j >= popsize) j = 0;
            if(fraction[j] > 0.0)
            {
                /* A winner if true */
                if(flip(fraction[j])) 
                {
                    choices[k] = j;
                    fraction[j] = fraction[j] - 1.0;
                    k++;
                }
            }
            j++;
        }
    }
    nremain = popsize - 1;
}


int select()
/* selection using remainder method */
{
    int jpick, slect;

    jpick = rnd(0, nremain);
    slect = choices[jpick];
    choices[jpick] = choices[nremain];
    nremain--;
    return(slect);
}
