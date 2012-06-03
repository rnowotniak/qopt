/***************************************************************************/
/*        ____Demonstrates passing a stream to another process____         */
/* Process 0 initializes a random number stream and prints a few random    */
/* numbers. It then passes this stream to process 1, which recieves it     */
/* and prints a few random numbers from this stream.                       */ 
/***************************************************************************/


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>		/* MPI header file                         */

/* Uncomment the following line to get the interface with pointer checking */
/*#define CHECK_POINTERS  */
 
#include "sprng.h"              /* SPRNG header file                       */

#define SEED 985456376



main(int argc, char *argv[])
{
  int streamnum, nstreams, *stream;
  double rn;
  int i, myid, nprocs, len;
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
  
  if (myid==0)	 /*********** process 0 sends stream to process 1 **********/
  {
    streamnum = 0;
    nstreams = 1;
    stream = init_sprng(streamnum,nstreams,SEED,SPRNG_DEFAULT);/*initialize stream*/
    printf("Process %d: Print information about stream:\n",myid);
    print_sprng(stream);

    printf("Process %d: Print 2 random numbers in [0,1):\n", myid);
    for (i=0;i<2;i++)
    {
      rn = sprng(stream);	/* generate double precision random number */
      printf("Process %d: %f\n", myid, rn);
    }

    len = pack_sprng(stream, &packed);   /* pack stream into an array      */
    /* inform process 1 how many bytes process 0 will send.                */
    MPI_Send(&len, 1, MPI_INT, 1, 0, MPI_COMM_WORLD); 
    MPI_Send(packed, len, MPI_BYTE, 1, 0, MPI_COMM_WORLD); /* send stream  */
    free(packed);		/* free storage for array                  */
    nstreams = free_sprng(stream);

    printf(" Process 0 sends stream to process 1\n");
    printf(" %d generators now exist on process 0\n", nstreams);
  }
  else if(myid == 1)  /***** process 1 receives stream from process 0 ******/
  {
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
    stream = unpack_sprng(packed); /* unpack stream                        */
    printf(" Process 1 has received the packed stream\n");
    printf("Process %d: Print information about stream:\n",myid);
    print_sprng(stream);
    free(packed);		/* free array of packed stream             */

    printf(" Process 1 prints 2 numbers from received stream:\n");
   for (i=0;i<2;i++)		
    {
      rn = sprng(stream);	/* generate double precision random number */
      printf("Process %d: %f\n", myid, rn);
    }

    free_sprng(stream);		/* free memory used to store stream state  */
  }

  MPI_Finalize();		/* terminate MPI                           */
}
