/*----------------------------------------------------------------------------*/
/* generate.c - create a new generation of individuals                        */
/*----------------------------------------------------------------------------*/

#include "external.h"

generation()
{
  
  int mate1, mate2, jcross, j = 0;
  
  /* perform any preselection actions necessary before generation */
  preselect();
  
  /* select, crossover, and mutation */
  do 
    {
      /* pick a pair of mates */
      mate1 = select(); 
      mate2 = select();
      
      /* Crossover and mutation */
      jcross = crossover(oldpop[mate1].chrom, oldpop[mate2].chrom,
			 newpop[j].chrom, newpop[j+1].chrom);
      mutation(newpop[j].chrom);
      mutation(newpop[j+1].chrom);
      
      /* Decode string, evaluate fitness, & record */
      /* parentage date on both children */
      objfunc(&(newpop[j]));
      newpop[j].parent[0] = mate1+1;
      newpop[j].xsite = jcross;
      newpop[j].parent[1] = mate2+1;
      objfunc(&(newpop[j+1]));
      newpop[j+1].parent[0] = mate1+1;
      newpop[j+1].xsite = jcross;
      newpop[j+1].parent[1] = mate2+1;
      
      /* Increment population index */
      j = j + 2;
    }
  while(j < (popsize-1));
  
}
