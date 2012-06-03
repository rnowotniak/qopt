/***************************************************************************/
/*         ____Demonstrates use of the single precision generator____      */
/* One stream is maintained per processor. Each processor prints a few     */
/* single precision random numbers.                                        */
/***************************************************************************/

#include <stdio.h>
#include <mpi.h>                /* MPI header file                         */


/* Uncomment the following line to get the interface with pointer checking */
/*#define CHECK_POINTERS                                                   */
 
#define FLOAT_GEN	  /* make 'sprng()' return single precision numbers*/
#include "sprng.h"              /* SPRNG header file                       */

#define SEED 985456376



main(int argc, char *argv[])
{
  int streamnum, nstreams, *stream;
  float rn;
  int i, myid, nprocs;


  /************************** MPI calls ************************************/
            
  MPI_Init(&argc, &argv);       /* Initialize MPI                          */
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);	/* find process id                 */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* find number of processes      */

  /************************* Initialization ********************************/
            
  streamnum = myid;
  nstreams = nprocs;		/* one stream per processor                */

  stream = init_sprng(streamnum,nstreams,SEED,SPRNG_DEFAULT);	/*initialize stream*/
  printf("Process %d: Print information about stream:\n",myid);
  print_sprng(stream);

  /*********************** print random numbers ****************************/
            
  for (i=0;i<3;i++)
  {
    rn = sprng(stream);		/* generate single precision random number */
    printf("Process %d, random number %d: %f\n", myid, i+1, rn);
  }

  /*************************** free memory *********************************/
            
  free_sprng(stream);           /* free memory used to store stream state  */

  MPI_Finalize();		/* Terminate MPI                           */
}
