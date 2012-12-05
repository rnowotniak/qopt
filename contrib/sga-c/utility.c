/*----------------------------------------------------------------------------*/
/* utility.c - utility routines, contains copyright,  repchar, skip           */
/*----------------------------------------------------------------------------*/

#include "external.h"

void copyright()
{
    void repchar(), skip();
    int iskip;
    int ll = 59;

    iskip = (LINELENGTH - ll)/2;
    skip(outfp,1);
    repchar(outfp," ",iskip); repchar(outfp,"-",ll); skip(outfp,1); 
    repchar(outfp," ",iskip);  
    fprintf(outfp,"|        SGA-C (v1.1) - A Simple Genetic Algorithm         |\n"); 
    repchar(outfp," ",iskip);  
    fprintf(outfp,"|     (c) David E. Goldberg 1986, All Rights Reserved      |\n"); 
    repchar(outfp," ",iskip);  
    fprintf(outfp,"|        C version by Robert E. Smith, U. of Alabama       |\n"); 
    repchar(outfp," ",iskip);  
    fprintf(outfp,"|    v1.1 modifications by Jeff Earickson, Boeing Company  |\n"); 
    repchar(outfp," ",iskip); repchar(outfp,"-",ll); skip(outfp,2); 
}


void repchar (outfp,ch,repcount)
/* Repeatedly write a character to stdout */
FILE *outfp;
char *ch;
int repcount;
{
    int j;

    for (j = 1; j <= repcount; j++) fprintf(outfp,"%s", ch);
}


void skip(outfp,skipcount)
/* Skip skipcount lines */
FILE *outfp;
int skipcount;
{
    int j;

    for (j = 1; j <= skipcount; j++) fprintf(outfp,"\n");
}


int ithruj2int(i,j,from)
/* interpret bits i thru j of a individual as an integer      */ 
/* j MUST BE greater than or equal to i AND j-i < UINTSIZE-1  */
/* from is a chromosome, represented as an array of unsigneds */
int i,j;
unsigned *from;
{
    unsigned mask, temp;
    int bound_flag;
    int iisin, jisin;
    int i1, j1, out;
  
    if(j < i)
    {
        fprintf(stderr,"Error in ithruj2int: j < i\n");
        exit(-1);
    }
    if(j-i+1 > UINTSIZE)
    {
        fprintf(stderr,"Error in ithruj2int: j-i+1 > UINTSIZE\n");
        exit(-1);
    }
  
    iisin = i/UINTSIZE;
    jisin = j/UINTSIZE;

    i1 = i - (iisin*UINTSIZE);
    j1 = j - (jisin*UINTSIZE);

	if(i1 == 0)
 	{
  		iisin = iisin-1;
  		i1 = i - (iisin*UINTSIZE);
 	};

	if(j1 == 0)
 	{ 
  		jisin = jisin-1;
  		j1 = j - (jisin*UINTSIZE);
 	};

    /* check if bits fall across a word boundary */    
    if(iisin == jisin)
        bound_flag = 0;
    else
        bound_flag = 1;

    if(bound_flag == 0)
    {
        mask = 1;
        mask = (mask<<(j1-i1+1))-1;
        mask = mask<<(i1-1);
        out = (from[iisin]&mask)>>(i1-1);
        return(out);
    }
    else
    {
        mask = 1;
        mask = (mask<<j1)-1;
        temp = from[jisin]&mask;
        mask = 1;
        mask = (mask<<(UINTSIZE-i1+1))-1;
        mask = mask<<(i1-1);
        out = ((from[iisin]&mask)>>(i1-1)) | (temp<<(UINTSIZE-i1+1));
        return(out);
    }
}





