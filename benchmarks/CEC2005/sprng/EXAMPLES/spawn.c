/***************************************************************************/
/*            ____Demonstrates the use of spawn_sprng____                  */
/* A random number stream is initialized and a few random numbers are      */
/* printed. Then two streams are spawned and a few numbers from one of them*/
/* is printed.                                                            */
/***************************************************************************/

#include <stdio.h>

/* Uncomment the following line to get the interface with pointer checking */
/*#define CHECK_POINTERS                                                   */
 
#include "sprng.h"  /* SPRNG header file                                   */

#define SEED 985456376



main()
{
  int streamnum, nstreams, *stream, **new;
  double rn;
  int i, irn, nspawned;

  /****************** Initialization values *******************************/
            
  streamnum = 0;
  nstreams = 1;

  stream = init_sprng(streamnum,nstreams,SEED,SPRNG_DEFAULT); /* initialize stream */
  printf(" Print information about stream:\n");
  print_sprng(stream);	

  /*********************** print random numbers ***************************/

  printf(" Printing 2 random numbers in [0,1):\n");
  for (i=0;i<2;i++)
  {
    rn = sprng(stream);		/* generate double precision random number*/
    printf("%f\n", rn);
  }

  /**************************** spawn streams *****************************/

  printf(" Spawned two streams\n");
  nspawned = 2;
  nspawned = spawn_sprng(stream,2,&new); /* spawn 2 streams               */
  if(nspawned != 2)
  {
    fprintf(stderr,"Error: only %d streams spawned\n", nspawned);
    exit(1);
  }
  printf(" Information on first spawned stream:\n");
  print_sprng(new[0]);
  printf(" Information on second spawned stream:\n");
  print_sprng(new[1]);
  

  printf(" Printing 2 random numbers from second spawned stream:\n");
  for (i=0;i<2;i++)
  {
    rn = sprng(new[1]);	/* generate a random number                       */
    printf("%f\n", rn);
  }

  /*************************** free memory ********************************/

  free_sprng(stream);  /* free memory used to store stream state          */
  free_sprng(new[0]);  /* free memory used to store stream state          */
  free_sprng(new[1]);  /* free memory used to store stream state          */
  free(new);
}
 
