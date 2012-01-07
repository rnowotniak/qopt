/***************************************************************************/
/*       ____Demonstrates SPRNG use for Monte Carlo integration____        */
/* Compute pi using Monte Carlo integration. Random points are generated   */
/* in a square of size 2. The value of pi is estimated as four times the   */
/* the proportion of samples that fall within a circle of unit radius.     */
/***************************************************************************/


#include <stdio.h>
#include <math.h>
#include <string.h>

#define SIMPLE_SPRNG		/* simple interface                        */
#include "sprng.h"

#define EXACT_PI 3.141592653589793238462643



main(argc,argv)
int argc;
char *argv[];
{
  int in, n, in_old, n_old;
  double pi, error, stderror, p=EXACT_PI/4.0;
  char filename[80];
  
  initialize(&n, &in_old, &n_old, filename);	/* read args & initialize  */
  
  in = count_in_circle(n);	/* count samples in circle                 */
  
  in += in_old;			/* # in circle, in all runs                */
  n += n_old;			/* # of samples, in all runs               */
  pi = (4.0*in)/n;		/* estimate pi                             */
  error = fabs(pi - EXACT_PI);	/* determine error                         */
  stderror = 4*sqrt(p*(1.0-p)/n); /* standard error                        */
  printf( "pi is estimated as %18.16f from %12d samples.\n", pi, n);
  printf( "\tError = %10.8g, standard error = %10.8g\n", error, stderror);

  save_state(filename, in, n);	/* check-point final state                 */
}



int count_in_circle(n)		/* count # of samples in circle            */
int n;
{
  int i, in;
  double x, y;
  
  for (i=in=0; i<n; i++)	/* find number of samples in circle        */
  {
    x = 2*sprng() - 1.0;	/* x coordinate                            */
    y = 2*sprng() - 1.0;	/* y coordinate                            */
    if (x*x + y*y < 1.0)	/* check if point (x,y) is in circle       */
      in++;
  }

  return in;
}


/* Read arguments and initialize stream                                    */
int initialize(n, in_old, n_old, filename)
int *n, *in_old, *n_old; 
char *filename;
{
  int seed, size, new_old;
  char buffer[MAX_PACKED_LENGTH];
  FILE *fp;
  
  printf("Enter 9 for a new run, or 2 for the continuation of an old run:\n");
  scanf("%d", &new_old);
  printf("Enter name of file to store/load state of the stream:\n");
  scanf("%s", filename);
  printf("Enter number of new samples:\n");
  scanf("%d", n);

  if(new_old == 9)		/* new set of runs                         */
  {
    seed = make_sprng_seed();	/* make seed from date/time information    */
    
    init_sprng(seed,CRAYLCG);	/* initialize stream                       */
    print_sprng();		/* print information abour stream          */

    *in_old = 0;
    *n_old = 0;
  }
  else				/* continue from previously stored state   */
  {
    fp = fopen(filename,"r");	/* open file                               */
    if(fp == NULL)
    {
      fprintf(stderr,"ERROR opening file %s\n", filename);
      exit(1);
    }
    
    fread(in_old,1,sizeof(int),fp); /* cumulative # in circle previously   */
    fread(n_old,1,sizeof(int),fp);  /* cumulative # of samples previously  */
    fread(&size,1,sizeof(int),fp);  /* size of stored stream state         */
    fread(buffer,1,size,fp);	/* copy stream state to buffer             */
    unpack_sprng(buffer);	/* retrieve state of the stream            */
    fclose(fp);			/* close file                              */
  }
  
  return 0;
}


int save_state(filename, in, n)	/* store the state                         */
char *filename;
int in, n;
{
  FILE *fp;
  char *bytes;
  int size;
  
  fp = fopen(filename,"w");	/* open file to store state                */
  if(fp == NULL)
  {
    fprintf(stderr,"Could not open file %s for writing\nCheck path or permissions\n", filename);
    exit(1);
  }

  fwrite(&in,1,sizeof(int),fp); /* store # in circle in all runs           */
  fwrite(&n,1,sizeof(int),fp);  /* store # of samples in all runs          */

  size = pack_sprng(&bytes);	/* pack stream state into an array         */
  fwrite(&size,1,sizeof(int),fp); /* store # of bytes required for storage */
  fwrite(bytes,1,size,fp);      /* store stream state                      */
  fclose(fp);

  free(bytes);			/* free memory needed to store stream state*/

  return 0;
}
