/*----------------------------------------------------------------------------*/
/* app.c - application dependent routines, change these for different problem */
/*                                                                            */
/* This example application interprets chromosomes as concatenated strings of */
/* binary integers of user-specified length (field_size).  Fitness is simply  */
/* the length of this integer vector squared.                                 */
/*----------------------------------------------------------------------------*/

#include "external.h"

static int field_size, vec_size;

application()
/* This routine should contain any application-dependent      */
/* computations that should be performed before each GA cycle */
/* called by main() */
{
}


app_data()
/* application dependent data input, called by init_data() */
/* ask your input questions here, and put output in global variables */
/* In this example application, the utility pointer of each individual is   */
/* assigned a vector of integers that will be used to store the interpreted */
/* chromosome. */
{
    int size = UINTSIZE;

    if(lchrom < UINTSIZE) size = lchrom;

    /* user must specify length of concatenated integers in the chromosome. */
    fprintf(outfp," Enter field size (must be less than %d) ->", size);
    fscanf(infp,"%d",&field_size);
    vec_size = lchrom/field_size;
    if((((float)lchrom/(float)field_size)-vec_size) > 0.0) vec_size++;
}


app_free()
/* This routine should free any memory allocated */
/* in the application-dependent routines, called by freeall() */
{
    int i;
    
    for(i = 0; i < popsize; i++)
    {
        free(newpop[i].utility);
        free(oldpop[i].utility);
    }
}


app_init()
/* Application dependent initialization routine called by initialize().       */
{
}

app_initreport()
/* Application-dependent initial report called by initialize() */
{
    if(vec_size > lchrom/field_size)
    {
        fprintf(outfp," Each chromosome interpreted as %d %d-bit integers",
            vec_size-1, field_size);
        fprintf(outfp," and one %d-bit integer.\n",lchrom-((vec_size-1)*field_size));
    }
    else
        fprintf(outfp,"Each chromosome interpreted as %d %d-bit integers.\n",
            vec_size, field_size);
}


app_malloc()
/* application dependent malloc() calls, called by initmalloc() */
{
    char *malloc();
    unsigned nbytes;
    int i;

    nbytes = vec_size * sizeof(int);
    for(i = 0; i < popsize; i++)
    {
        if((newpop[i].utility = (int *) malloc(nbytes)) == NULL)
            nomemory(stderr,"newpop utility");
        if((oldpop[i].utility = (int *) malloc(nbytes)) == NULL)
            nomemory(stderr,"oldpop utility");
    }
}


app_report()
/* Application-dependent report, called by report() */
{
    int i, j;

    /* Print vector interpretation of ech chromosome. */   
    for(i = 0; i < popsize; i++)
    {
        fprintf(outfp,"oldpop %d = ", i);
        for(j = 0; j < vec_size; j++)
            fprintf(outfp, "%d ", oldpop[i].utility[j]);
        fprintf(outfp,"\n");
    }
}


app_stats(pop)
/* Application-dependent statistics calculations called by statistic() */
struct individual *pop;
{
}


objfunc(critter)
/* Application dependent objective function */
struct individual *critter;
{
    int i, start, stop;

    /* Interpret each chromosome as a vector of concatenanted integers */
    critter->fitness = 0;

    for(i = 0; i < vec_size; i++)
    {
        /* section of chromosome containing current integer field */
        start = (i * field_size) + 1;
        stop  = ((i + 1) * field_size);

        /* check if enough bits remain, if not, interpret as a short field */
        if(stop > lchrom) stop = lchrom;

        /* convert bit field in chromosome to an integer    */
        /* and store it in utility array.  Then compute     */
        /* chromosome fitness as sum of squares of integers */
        critter->utility[i] = ithruj2int(start, stop, critter->chrom);
        critter->fitness += critter->utility[i]*critter->utility[i];
    }
}
