/****************************************************************************/
/*               ____Demonstrates the use make_sprng_seed____               */
/* 'make_sprng_seed' is used to produce a new seed each time the program is */
/* run. Then a few random numbers are printed.                              */
/****************************************************************************/

#include <stdio.h>

/* Uncomment the following line to get the interface with pointer checking */
/*#define CHECK_POINTERS                                                   */
 
#include "sprng.h"              /* SPRNG header file                       */

main()
{
  int streamnum, nstreams, seed, *stream, i;
  double rn;



  /************************** Initialization *******************************/

  streamnum = 0;
  nstreams = 1;

  seed = make_sprng_seed();	/* make new seed each time program is run  */

  stream = init_sprng(streamnum,nstreams,seed,SPRNG_DEFAULT); /*initialize stream*/
  printf(" Printing information about new stream\n");
  print_sprng(stream);

  /************************ print random numbers ***************************/

  printf(" Printing 3 random numbers in [0,1):\n");
  for (i=0;i<3;i++)
  {
    rn = sprng(stream);		/* generate double precision random number */
    printf("%f\n", rn);
  }

  free_sprng(stream);		/* free memory used to store stream state  */
}
