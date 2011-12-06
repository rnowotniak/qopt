/***************************************************************************/
/*            ____Demonstrates the use of sprng and isprng____             */
/* A random number stream is initialized and a few random double precision */
/* numbers and a few integers are printed.                                 */
/***************************************************************************/

#include <stdio.h>

/* Uncomment the following line to get the interface with pointer checking */
/*#define CHECK_POINTERS                                                   */
 
#include "sprng.h"  /* SPRNG header file                                   */

#define SEED 985456376



main()
{
  int streamnum, nstreams, *stream;
  double rn;
  int irn;
  int i;

  /****************** Initialization values *******************************/
            
  streamnum = 0;
  nstreams = 1;

  stream = init_sprng(streamnum,nstreams,SEED,SPRNG_DEFAULT); /* initialize stream */
  printf(" Print information about new stream:\n");
  print_sprng(stream);	

  /*********************** print random numbers ***************************/

  printf(" Printing 3 random numbers in [0,1):\n");
  for (i=0;i<3;i++)
  {
    rn = sprng(stream);		/* generate a double precision random number */
    printf("%f\n",rn);
  }

  printf(" Printing 3 random integers in [0,2^31):\n");
  for (i=0;i<3;i++)
  {
    irn = isprng(stream);	/* generate an integer random number */
    printf("%16d\n",irn);
  }

  /*************************** free memory ********************************/

  free_sprng(stream);  /* free memory used to store stream state */

}
