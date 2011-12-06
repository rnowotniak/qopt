/***************************************************************************/
/*         ____Demonstrates use of shared and non-shared streams____       */
/* Each process has two streams.  One stream is common to all the          */
/* processes. The other stream is different on each process.               */
/***************************************************************************/

#include <stdio.h>
#include <mpi.h>                 /* MPI header file                        */

/* Uncomment the following line to get the interface with pointer checking */
/*#define CHECK_POINTERS                                                   */
 
#include "sprng.h"               /* SPRNG header file                      */

#define SEED 985456376



main(int argc, char *argv[])
{
  int streamnum, commNum, nstreams, *stream, *commonStream;
  double rn;
  int i, myid, nprocs;


  /************************** MPI calls ***********************************/
            
  MPI_Init(&argc, &argv);       /* Initialize MPI                         */
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);	/* find process id                */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* find number of processes     */

  /****************** Initialization values *******************************/
            
  streamnum = myid;		/*This stream is different on each process*/
  commNum = nprocs;	        /* This stream is common to all processes */
  nstreams = nprocs + 1;	/* extra stream is common to all processes*/

  /*********************** Initialize streams *****************************/
            
  /* This stream is different on each process                             */
  stream = init_sprng(streamnum,nstreams,SEED,SPRNG_DEFAULT);
  printf("Process %d: Print information about new stream\n", myid);
  print_sprng(stream);

  /* This stream is identical on each process                             */
  commonStream = init_sprng(commNum,nstreams,SEED,SPRNG_DEFAULT);
  printf("Process %d: This stream is identical on all processes\n", myid);
  print_sprng(commonStream);

  /*********************** print random numbers ***************************/
            
  for (i=0;i<2;i++)		/* random numbers from distinct stream    */
  {
    rn = sprng(stream);		/* generate double precision random number*/
    printf("Process %d, random number (distinct stream) %d: %f\n",
	   myid, i+1, rn);
  }

  for (i=0;i<2;i++)		/* random number from common stream       */
  {
    rn = sprng(commonStream);	/*generate double precision random number */
    printf("Process %d, random number (shared stream) %d: %f\n", myid, i+1, rn);
  }

  /*************************** free memory ********************************/
            
  free_sprng(stream);          /* free memory used to store stream state  */
  free_sprng(commonStream);

  MPI_Finalize();              /* terminate MPI                           */

}
