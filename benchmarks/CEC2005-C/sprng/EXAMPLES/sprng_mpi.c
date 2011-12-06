/***************************************************************************/
/*           Demonstrates sprng use with one stream per process            */
/* A distinct stream is created on each process, then prints a few         */
/* random numbers.                                                         */
/***************************************************************************/


#include <stdio.h>
#include <mpi.h>		/* MPI header file                         */

/* Uncomment the following line to get the interface with pointer checking */
/*#define CHECK_POINTERS                                                   */
 
#include "sprng.h"		/* SPRNG header file                       */

#define SEED 985456376



main(int argc, char *argv[])
{
  int streamnum, nstreams, *stream;
  double rn;
  int i, myid, nprocs;


  /*************************** MPI calls ***********************************/

  MPI_Init(&argc, &argv);	/* Initialize MPI                          */
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);	/* find process id                 */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* find number of processes      */

  /************************** Initialization *******************************/

  streamnum = myid;	
  nstreams = nprocs;		/* one stream per processor                */

  stream = init_sprng(streamnum,nstreams,SEED,SPRNG_DEFAULT);	/* initialize stream */
  printf("Process %d, print information about stream:\n", myid);
  print_sprng(stream);

  /*********************** print random numbers ****************************/

  for (i=0;i<3;i++)
  {
    rn = sprng(stream);		/* generate double precision random number */
    printf("Process %d, random number %d: %.14f\n", myid, i+1, rn);
  }

  /*************************** free memory *********************************/

  free_sprng(stream);		/* free memory used to store stream state  */

  MPI_Finalize();		/* Terminate MPI                           */
}
