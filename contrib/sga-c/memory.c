/*----------------------------------------------------------------------------*/
/* memory.c - memory management routines for sga code                         */
/*----------------------------------------------------------------------------*/

#include "external.h"

initmalloc()
     /* memory allocation of space for global data structures */
{
  unsigned nbytes;
  char     *malloc();
  int j;
  
  /* memory for old and new populations of individuals */
  nbytes = popsize*sizeof(struct individual);
  if((oldpop = (struct individual *) malloc(nbytes)) == NULL)
    nomemory(stderr,"oldpop");
  
  if((newpop = (struct individual *) malloc(nbytes)) == NULL)
    nomemory(stderr,"newpop");
  
  /* memory for chromosome strings in populations */
  nbytes = chromsize*sizeof(unsigned);
  for(j = 0; j < popsize; j++)
    {
      if((oldpop[j].chrom = (unsigned *) malloc(nbytes)) == NULL)
	nomemory(stderr,"oldpop chromosomes");
      
      if((newpop[j].chrom = (unsigned *) malloc(nbytes)) == NULL)
	nomemory(stderr,"newpop chromosomes");
    }
  
  if((bestfit.chrom = (unsigned *) malloc(nbytes)) == NULL)
    nomemory(stderr,"bestfit chromosome");

  /* allocate any auxiliary memory for selection */
  select_memory();
  
  /* call to application-specific malloc() routines   */
  /* can be used to malloc memory for utility pointer */
  app_malloc();
}


freeall()
     /* A routine to free all the space dynamically allocated in initspace() */
{
  int i;
  
  for(i = 0; i < popsize; i++)
    {  
      free(oldpop[i].chrom);
      free(newpop[i].chrom);
    }
  free(oldpop);
  free(newpop);
  free(bestfit.chrom);
  
  /* free any auxiliary memory needed for selection */
  select_free();
  
  /* call to application-specific free() routines   */
  /* can be used to free memory for utility pointer */
  app_free();
}


nomemory(string)
     char *string;
{
  fprintf(outfp,"malloc: out of memory making %s!!\n",string);
  exit(-1);
}


