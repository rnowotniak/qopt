/*----------------------------------------------------------------------------*/
/* external.h - external global declarations from sga.h.                      */
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

extern struct individual *oldpop;           /* last generation of individuals */
extern struct individual *newpop;           /* next generation of individuals */
extern struct bestever bestfit;                  /* fittest individual so far */
extern double sumfitness;             /* summed fitness for entire population */
extern double max;                           /* maximum fitness of population */
extern double avg;                           /* average fitness of population */
extern double min;                           /* minumum fitness of population */
extern float  pcross;                             /* probability of crossover */
extern float  pmutation;                           /* probability of mutation */
extern int    numfiles;                               /* number of open files */
extern int    popsize;                                     /* population size */
extern int    lchrom;              /* length of the chromosome per individual */
extern int    chromsize;     /* number of bytes needed to store lchrom string */
extern int    gen;                               /* current generation number */
extern int    maxgen;                            /* maximum generation number */
extern int    run;                                      /* current run number */
extern int    maxruns;                      /* maximum number of runs to make */
extern int    printstrings;  /* flag to print chromosome strings (default on) */
extern int    nmutation;                      /* number of mutations  */
extern int    ncross;                        /* number of crossovers  */

/* Application-dependent external declarations go after here...  */
