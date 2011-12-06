/***************************************************************************/
/*           Demonstrates sprng use with one stream per process            */
/* A distinct stream is created on each process, then prints a few         */
/* random numbers.                                                         */
/***************************************************************************/


#include <stdio.h>
#include <mpi.h>                /* MPI header file                         */

#define SIMPLE_SPRNG		/* simple interface                        */
#define USE_MPI			/* use MPI to find number of processes     */
#include "sprng.h"              /* SPRNG header file                       */

#define SEED 985456376



main(int argc, char *argv[])
{
  double rn;
  int i, myid;


  /*************************** MPI calls ***********************************/
            
  MPI_Init(&argc, &argv);       /* Initialize MPI                          */
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);	/* find process id                 */

  /************************** Initialization *******************************/

  init_sprng(SEED,SPRNG_DEFAULT);	/* initialize stream                       */
  printf("Process %d, print information about stream:\n", myid);
  print_sprng();

  /************************ print random numbers ***************************/
            
  for (i=0;i<3;i++)
  {
    rn = sprng();		/* generate double precision random number */
    printf("Process %d, random number %d: %.14f\n", myid, i+1, rn);
  }

  MPI_Finalize();		/* Terminate MPI                           */
}
