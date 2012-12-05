/*******************************************/
/*      Simple Genetic Algorithm - SGA     */
/*           Haploid Version               */
/* (c)   David Edward Goldberg  1986       */
/*          All Rights Reserved            */
/*      C translation by R.E. Smith        */
/*  v1.1 modifications by Jeff Earickson   */
/*******************************************/

#include "sga.h"

main(argc,argv)
int argc;
char *argv[];
{
    struct individual *temp;
    FILE   *fopen();
    void   copyright();
    char   *malloc();


    /* determine input and output from program args */
    numfiles = argc - 1;
    switch(numfiles)
    {
        case 0:
            infp = stdin;
            outfp = stdout;
            break;
        case 1:
            if((infp = fopen(argv[1],"r")) == NULL)
            {
                fprintf(stderr,"Input file %s not found\n",argv[1]);
                exit(-1);
            }
            outfp = stdout;
            break;
        case 2:
            if((infp = fopen(argv[1],"r")) == NULL)
            {
                fprintf(stderr,"Cannot open input file %s\n",argv[1]);
                exit(-1);
            }
            if((outfp = fopen(argv[2],"w")) == NULL)
            {
                fprintf(stderr,"Cannot open output file %s\n",argv[2]);
                exit(-1);
            }
            break;
        default:
            fprintf(stderr,"Usage is: sga [input file] [output file]\n");
            exit(-1);
    }


    /* print the author/copyright notice */
    copyright();
 
    if(numfiles == 0)
        fprintf(outfp," Number of GA runs to be performed-> ");
    fscanf(infp,"%d",&maxruns);

    for(run=1; run<=maxruns; run++)
    {
        /* Set things up */
        initialize();
        
        for(gen=0; gen<maxgen; gen++)
        {
            fprintf(outfp,"\nRUN %d of %d: GENERATION %d->%d\n",
                           run,maxruns,gen,maxgen);

            /*application dependent routines*/
            application();

            /* create a new generation */
            generation();


            /* compute fitness statistics on new populations */
            statistics(newpop);

            /* report results for new generation */
            report();

            /* advance the generation */
            temp = oldpop;
            oldpop = newpop;
            newpop = temp;
        }
        freeall();
    }

}


