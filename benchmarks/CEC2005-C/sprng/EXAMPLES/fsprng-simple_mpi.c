/***************************************************************************/
/*         ____Demonstrates use of the single precision generator____      */
/* One stream is maintained per processor. Each processor prints a few     */
/* single precision random numbers.                                        */
/***************************************************************************/

#include <stdio.h>
#include <mpi.h>                /* MPI header file                         */

#define SIMPLE_SPRNG		/* simple interface                        */
#define USE_MPI			/* use MPI to find number of processes     */
#define FLOAT_GEN	  /* make 'sprng()' return single precision numbers*/
#include "sprng.h"              /* SPRNG header file                       */

#define SEED 985456376



main(int argc, char *argv[])
{
  int i, myid;
  float rn;


  /************************** MPI calls ***********************************/
            
  MPI_Init(&argc, &argv);       /* Initialize MPI                         */
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);	/* find process id                */

  /*********************** Initialize streams *****************************/

  init_sprng(SEED,SPRNG_DEFAULT);	/* initialize stream                      */
  printf("Process %d: Print information about stream:\n",myid);
  print_sprng();

  /*********************** print random numbers ***************************/
            
  for (i=0;i<3;i++)
  {
    rn = sprng();		/*generate single precision random number */
    printf("Process %d, random number %d: %f\n", myid, i+1, rn);
  }

  MPI_Finalize();		/* Terminate MPI */
}
