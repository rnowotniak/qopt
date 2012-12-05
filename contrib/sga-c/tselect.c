/*----------------------------------------------------------------------------*/
/* tselect.c - Tournament selection                                           */
/* Contributed by Hillol Kargupta, Dept of Engr. Mechanics, U. of Alabama     */
/*----------------------------------------------------------------------------*/

#include "external.h"   

static int    *tourneylist, tourneypos, tourneysize;  
/* tournament list, position in list */

select_memory()
{
  unsigned nbytes;
  char     *malloc();
  int j;
  
  nbytes = popsize*sizeof(int);
  if((tourneylist = (int *) malloc(nbytes)) == NULL)
    nomemory(stderr,"tourneylist");
  
  if(numfiles == 0)
    fprintf(outfp," Enter tournament size for selection --> ");
  fscanf(infp,"%d", &tourneysize);
  if(tourneysize > popsize)
    {
      fprintf(outfp,"FATAL: Tournament size (%d) > popsize (%d)\n",
	      tourneysize,popsize);
      exit(-1);
    }
  
}

select_free()
{
  free(tourneylist);
};


  

preselect()
{
    reset();
    tourneypos = 0;
}


int select()
{
    int pick, winner, i;

    /* If remaining members not enough for a tournament, then reset list */
    if((popsize - tourneypos) < tourneysize)
    {
        reset();
        tourneypos = 0;
    }

    /* Select tourneysize structures at random and conduct a tournament */
    winner=tourneylist[tourneypos];
    for(i=1; i<tourneysize; i++)
    {
        pick=tourneylist[i+tourneypos];
        if(oldpop[pick].fitness > oldpop[winner].fitness) winner=pick;
    }

    /* Update tourneypos */
    tourneypos += tourneysize; 
    return(winner);
}


reset()   
/* Shuffles the tourneylist at random */
{
    int i, rand1, rand2, temp;

    for(i=0; i<popsize; i++) tourneylist[i] = i;

    for(i=0; i < popsize; i++)
    {
        rand1=rnd(i,popsize-1);
        rand2=rnd(i,popsize-1);
        temp = tourneylist[rand1];
        tourneylist[rand1]=tourneylist[rand2];
        tourneylist[rand2]=temp;
    }
}
