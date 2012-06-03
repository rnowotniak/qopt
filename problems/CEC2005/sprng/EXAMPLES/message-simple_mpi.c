/***************************************************************************/
/*        ____Demonstrates passing a stream to another process____         */
/* Process 0 initializes a random number stream and prints a few random    */
/* numbers. It then passes this stream to process 1, which recieves it     */
/* and prints a few random numbers from this stream.                       */ 
/***************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>		/* MPI header file                         */

#define SIMPLE_SPRNG		/* simple interface                        */
#include "sprng.h"              /* SPRNG header file                       */

#define SEED 985456376



main(int argc, char *argv[])
{
  double rn;
  int i, myid, len, nprocs;
  MPI_Status  status;
  char *packed;


  /************************** MPI calls ************************************/
            
  MPI_Init(&argc, &argv);	/* Initialize MPI                          */
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);	/* find process id                 */
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* find number of processes      */

  if(nprocs < 2)
  {
    fprintf(stderr,"ERROR: At least 2 processes required\n");
    MPI_Finalize();
    exit(1);
  }
  
  if (myid==0)	/*********** process 0 sends stream to process 1 ***********/
  {
    init_sprng(SEED,SPRNG_DEFAULT);	/*initialize stream                        */
    printf("Process %d: Print information about stream:\n",myid);
    print_sprng();

    printf("Process %d: Print 2 random numbers in [0,1):\n", myid);
    for (i=0;i<2;i++)
    {
      rn = sprng();		/* generate double precision random number */
      printf("Process %d: %f\n", myid, rn);
    }

    len = pack_sprng(&packed);	/* pack stream into an array               */
    /* inform process 1 how many bytes process 0 will send.                */
    MPI_Send(&len, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); 
    MPI_Send(packed, len, MPI_BYTE, 1, 0, MPI_COMM_WORLD); /* send stream  */
    free(packed);		/* free storage for array                  */
    printf(" Process 0 sends stream to process 1\n");
  }
  else if(myid == 1)  /****** process 1 receives stream from process 0 *****/
  {
    init_sprng(SEED,SPRNG_DEFAULT);   /*initialize stream                        */
    MPI_Recv(&len, 1, MPI_INT, 0, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status); /* receive buffer size required     */
    
    if ((packed = (char *) malloc(len)) == NULL) /* allocate array         */
    {
      fprintf(stderr,"ERROR: process %d: Cannot allocate memory\n", myid);
      MPI_Finalize();
      exit(1);
    }

    MPI_Recv(packed, len, MPI_BYTE, 0, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status); /* receive packed stream            */
    unpack_sprng(packed);	/* unpack stream                           */
    printf(" Process 1 has received the packed stream\n");
    printf("Process %d: Print information about stream:\n",myid);
    print_sprng();
    free(packed);		/* free array of packed stream             */

    printf(" Process 1 prints 2 numbers from received stream:\n");
   for (i=0;i<2;i++)		
    {
      rn = sprng();		/* generate double precision random number */
      printf("Process %d: %f\n", myid, rn);
    }

  }

  MPI_Finalize();		/* terminate MPI                           */
}
