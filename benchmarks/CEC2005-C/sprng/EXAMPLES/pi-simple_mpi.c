/***************************************************************************/
/*       ____Demonstrates SPRNG use for Monte Carlo integration____        */
/* Compute pi using Monte Carlo integration. Random points are generated   */
/* in a square of size 2. The value of pi is estimated as four times the   */
/* the proportion of samples that fall within a circle of unit radius.     */
/* The final state of the computations is check-pointed at the end, so the */
/* the calculations can be continued from the previous state. However, the */
/* same number of processors should be used as in the previous run.        */
/***************************************************************************/


#include <stdio.h>
#include <math.h>
#include <string.h>
#include "mpi.h"

#define SIMPLE_SPRNG		/* simple interface                        */
#define USE_MPI			/* MPI version of SPRNG                    */
#include "sprng.h"

#define EXACT_PI 3.141592653589793238462643
#define RECV_STREAM_TAG 1

main(argc,argv)
int argc;
char *argv[];
{
  int in, i, seed, n, my_n, in_old, n_old, nprocs, myid, temp;
  double pi, error, stderror, p=EXACT_PI/4.0;
  char filename[80];
  
  /*************************** Initialization ******************************/

  MPI_Init(&argc,&argv);	/* Initialize MPI                          */
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs); /* Find number of processes       */
  MPI_Comm_rank(MPI_COMM_WORLD,&myid); /* Find rank of process             */

  initialize(&n, &in_old, &n_old, filename);	/* read args & initialize  */
  
  my_n = n/nprocs;		/* number of samples for this process      */
  if(myid < n%nprocs)
    my_n++;

  /******************** Count number of samples in circle ******************/

  temp = count_in_circle(my_n);	/* count samples in circle                 */
  
				/* sum # in circle over all processes      */
  MPI_Reduce(&temp, &in, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  /*************************** Determine Pi ********************************/

  if(myid == 0)
  {
    in += in_old;		/* # in circle, in all runs                */
    n += n_old;			/* # of samples, in all runs               */
    pi = (4.0*in)/n;		/* estimate pi                             */
    error = fabs(pi - EXACT_PI); /* determine error                        */
    stderror = 4*sqrt(p*(1.0-p)/n); /* standard error                      */
    printf( "pi is estimated as %18.16f from %12d samples.\n", pi, n);
    printf( "\tError = %10.8g, standard error = %10.8g\n", error, stderror);
  }
  
  /*************************** Save final state ****************************/

  save_state(filename,in,n); /* save final state of computation*/

  MPI_Finalize();		/* Terminate MPI                           */
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
  int seed, size, myid, nprocs, new_old, old_nprocs, i;
  char buffer[MAX_PACKED_LENGTH];
  FILE *fp;
  MPI_Status status;
  
  MPI_Comm_rank(MPI_COMM_WORLD,&myid); /* find rank of process             */
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs); /* Find number of processes       */

  if(myid == 0)
  {
    printf("Enter 9 for a new run, or 2 to continue an old run:\n");
    scanf("%d", &new_old);
    printf("Enter name of file to store/load state of the stream:\n");
    scanf("%s", filename);
    printf("Enter number of new samples:\n");
    scanf("%d", n);
  }
  
  MPI_Bcast(&new_old,1,MPI_INT,0,MPI_COMM_WORLD);
  MPI_Bcast(n,1,MPI_INT,0,MPI_COMM_WORLD);
  
    
  if(new_old == 9)		/* new set of runs                         */
  {
      seed = make_sprng_seed();	/* make seed from date/time information    */
    
      init_sprng(seed,CRAYLCG);	/* initialize stream                       */
      print_sprng();

    *in_old = 0;
    *n_old = 0;
  }
  else				/* continue from previously stored state   */
  {
    if(myid == 0)
    {
      fp = fopen(filename,"r");	/* open file                               */
      if(fp == NULL)
      {
	fprintf(stderr,"ERROR opening file %s\nPlease kill all processes\n", 
		filename);
	exit(1);
      }
    
      fread(in_old,1,sizeof(int),fp); /* cumulative # in circle previously   */
      fread(n_old,1,sizeof(int),fp);  /* cumulative # of samples previously  */
      fread(&old_nprocs,1,sizeof(int),fp);/*cumulative # of previous samples */

      if(old_nprocs != nprocs)
	fprintf(stderr,"Number of processes different in current and previous\
 runs. Strange thing can happen ... \n");
      
      fread(&size,1,sizeof(int),fp);  /* size of stored stream state         */

      fread(buffer,1,size,fp);	/* copy stream state to buffer             */
      unpack_sprng(buffer);	/* retrieve state of the stream            */

      for(i=1; i<old_nprocs; i++)
      {
	fread(buffer,1,size,fp); /* copy stream state to buffer            */
	MPI_Send(buffer, size, MPI_BYTE, i, RECV_STREAM_TAG, MPI_COMM_WORLD); 
      }
      
      fclose(fp);		/* close file                              */
    }
    else
    {
      MPI_Recv(buffer, MAX_PACKED_LENGTH, MPI_BYTE, 0, RECV_STREAM_TAG,
	       MPI_COMM_WORLD, &status); /* receive packed stream          */
      unpack_sprng(buffer);	/* unpack stream                           */
    }
  }
  
  return 0;
}


int save_state(filename, in, n)	/* store the state                        */
char *filename;
int in, n;
{
  FILE *fp;
  char *bytes, packed[MAX_PACKED_LENGTH];
  int size, myid, nprocs, i;
  MPI_Status status;
  
  MPI_Comm_rank(MPI_COMM_WORLD,&myid); /* Find process rank                */
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs); /* Find number of processes       */

  if(myid == 0)
  {
    fp = fopen(filename,"w");	/* open file to store stream state         */
    if(fp == NULL)
    {
      fprintf(stderr,"Could not open file %s for writing\nCheck path or permissions\n", filename);
      exit(1);
    }

    fwrite(&in,1,sizeof(int),fp); /* store # in circle in all runs         */
    fwrite(&n,1,sizeof(int),fp);  /* store # of samples in all runs        */
    fwrite(&nprocs,1,sizeof(int),fp);  /* store # of processes             */

    size = pack_sprng(&bytes);	  /* pack stream state into an array       */
    fwrite(&size,1,sizeof(int),fp); /* store # of bytes for storage        */
    fwrite(bytes,1,size,fp);      /* store stream state                    */

    for(i=1; i<nprocs; i++)
    {
      MPI_Recv(packed, size, MPI_BYTE, i, RECV_STREAM_TAG,
	       MPI_COMM_WORLD, &status); /* receive packed stream          */
      fwrite(packed,1,size,fp);    /* store stream state                   */
    }
    
    fclose(fp);
  }
  else
  {
    size = pack_sprng(&bytes);	/* pack stream into an array               */
    MPI_Send(bytes, size, MPI_BYTE, 0, RECV_STREAM_TAG, MPI_COMM_WORLD); 
  }
  
  free(bytes);		  /* free memory needed to store state     */

  return 0;
}
