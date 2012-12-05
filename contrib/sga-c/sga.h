/*----------------------------------------------------------------------------*/
/* sga.h - global declarations for main(), all variable declared herein must  */
/*         also be defined as extern variables in external.h !!!              */
/*----------------------------------------------------------------------------*/

#define LINELENGTH 80                                    /* width of printout */
#define BITS_PER_BYTE 8            /* number of bits per byte on this machine */
#define UINTSIZE (BITS_PER_BYTE*sizeof(unsigned))    /* # of bits in unsigned */
#include <stdio.h>

/* file pointers */
FILE *outfp, *infp;

/* Global structures and variables */
struct individual 
{
    unsigned *chrom;                  /* chromosome string for the individual */
    double   fitness;                            /* fitness of the individual */
    int      xsite;                               /* crossover site at mating */
    int      parent[2];                  /* who the parents of offspring were */
    int      *utility;           /* utility field can be used as pointer to a */
                /* dynamically allocated, application-specific data structure */
};
struct bestever
{
    unsigned *chrom;        /* chromosome string for the best-ever individual */
    double   fitness;                  /* fitness of the best-ever individual */
    int      generation;                      /* generation which produced it */
};

struct individual *oldpop;                  /* last generation of individuals */
struct individual *newpop;                  /* next generation of individuals */
struct bestever bestfit;                         /* fittest individual so far */
double sumfitness;                    /* summed fitness for entire population */
double max;                                  /* maximum fitness of population */
double avg;                                  /* average fitness of population */
double min;                                  /* minumum fitness of population */
float  pcross;                                    /* probability of crossover */
float  pmutation;                                  /* probability of mutation */
int    numfiles;                                      /* number of open files */
int    popsize;                                            /* population size */
int    lchrom;                     /* length of the chromosome per individual */
int    chromsize;            /* number of bytes needed to store lchrom string */
int    gen;                                      /* current generation number */
int    maxgen;                                   /* maximum generation number */
int    run;                                             /* current run number */
int    maxruns;                             /* maximum number of runs to make */
int    printstrings = 1;     /* flag to print chromosome strings (default on) */
int    nmutation;                             /* number of mutations */
int    ncross;                               /* number of crossovers */

/* Application-dependent declarations go after here... */
