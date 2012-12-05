/*----------------------------------------------------------------------------*/
/* report.c - generation report files                                         */
/*----------------------------------------------------------------------------*/

#include "external.h"

report()
/* Write the population report */
{
    void   repchar(), skip();
    int    writepop(), writestats();

    repchar(outfp,"-",LINELENGTH); 
    skip(outfp,1);
    if(printstrings == 1)
    {
        repchar(outfp," ",((LINELENGTH-17)/2));
        fprintf(outfp,"Population Report\n");
        fprintf(outfp, "Generation %3d", gen);
        repchar(outfp," ",(LINELENGTH-28)); 
        fprintf(outfp, "Generation %3d\n", (gen+1));
        fprintf(outfp,"num   string ");
        repchar(outfp," ",lchrom-5);
        fprintf(outfp,"fitness    parents xsite  ");
        fprintf(outfp,"string ");
        repchar(outfp," ",lchrom-5);
        fprintf(outfp,"fitness\n");
        repchar(outfp,"-",LINELENGTH);
        skip(outfp,1);

        writepop(outfp);
        repchar(outfp,"-",LINELENGTH); 
        skip(outfp,1);

    }

    /* write the summary statistics in global mode  */
    fprintf(outfp,"Generation %d Accumulated Statistics: \n",
            gen);

    fprintf(outfp,"Total Crossovers = %d, Total Mutations = %d\n",
                   ncross,nmutation);
    fprintf(outfp,"min = %f   max = %f   avg = %f   sum = %f\n",
                 min,max,avg,sumfitness);
    fprintf(outfp,"Global Best Individual so far, Generation %d:\n",
                 bestfit.generation);
    fprintf(outfp,"Fitness = %f: ", bestfit.fitness);
    writechrom((&bestfit)->chrom);
    skip(outfp,1);
    repchar(outfp,"-",LINELENGTH);
    skip(outfp,1);


    /* application dependent report */
    app_report(); 
}




writepop()
{
    struct individual *pind;
    int j;

    for(j=0; j<popsize; j++)
    {
        fprintf(outfp,"%3d)  ",j+1);
   
        /* Old string */
        pind = &(oldpop[j]);
        writechrom(pind->chrom);
        fprintf(outfp,"  %8f | ", pind->fitness);
   
        /* New string */
        pind = &(newpop[j]);
        fprintf(outfp,"(%2d,%2d)   %2d   ",
        pind->parent[0], pind->parent[1], pind->xsite);
        writechrom(pind->chrom);
        fprintf(outfp,"  %8f\n", pind->fitness);
    }
}


writechrom(chrom)
/* Write a chromosome as a string of ones and zeroes            */
/* note that the most significant bit of the chromosome is the  */
/* RIGHTMOST bit, not the leftmost bit, as would be expected... */
unsigned *chrom;
{
    int j, k, stop;
    unsigned mask = 1, tmp;

    for(k = 0; k < chromsize; k++)
    {
        tmp = chrom[k];
        if(k == (chromsize-1))
            stop = lchrom - (k*UINTSIZE);
        else
            stop = UINTSIZE;

        for(j = 0; j < stop; j++)
        {
            if(tmp&mask)
                fprintf(outfp,"1");
            else
                fprintf(outfp,"0");
            tmp = tmp>>1;
        }
    }
}


