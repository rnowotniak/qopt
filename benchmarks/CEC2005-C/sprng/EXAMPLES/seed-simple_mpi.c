/***************************************************************************/
/*       ____Demonstrates the use of make_seed with MPI____                */
/* 'make_sprng_seed' is called to produce a new seed each time the program */
/* is run. The same seed is produced on each process.                      */
/***************************************************************************/

#include <stdio.h>
#include <mpi.h>                /* MPI header file                         */

#define SIMPLE_SPRNG		/* simple interface                        */
#define USE_MPI                 /* SPRNG makes MPI calls                   */
#include "sprng.h"              /* SPRNG header file                       */


main(int argc, char *argv[])
{
  int seed;
  double rn;
  int myid, i;



  /*************************** MPI calls ***********************************/
            
  MPI_Init(&argc, &argv);      /* Initialize MPI */
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);	/* find process id */

  /************************** Initialization *******************************/

  seed = make_sprng_seed();	/* make new seed each time program is run  */

  /* Seed should be the same on all processes                              */
  printf("Process %d: seed = %16d\n", myid, seed);

  init_sprng(seed,SPRNG_DEFAULT);	/* initialize stream                       */
  printf("Process %d: Print information about stream:\n",myid);
  print_sprng();

  /************************ print random numbers ***************************/

  for (i=0;i<3;i++)
  {
    rn = sprng();		/* generate double precision random number */
    printf("process %d, random number %d: %f\n", myid, i+1, rn);
  }

  MPI_Finalize();		/* Terminate MPI                           */
}
