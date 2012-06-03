/****************************************************************************/
/* ___Demonstrates invalid ID handling in inteface with pointer checking___ */
/* This prorgam prints a few random numbers, frees the stream, and then     */
/* tries to use the stream again.                                           */
/****************************************************************************/


#include <stdio.h>

#ifndef CHECK_POINTERS
#define CHECK_POINTERS		/* do not uncomment this line               */
#endif

#include "sprng.h"  /* SPRNG header file                                    */

#define SEED 985456376



main()
{
  int streamnum, nstreams, *stream;
  double rn;
  int i;



  /**************************** Initialize  *********************************/
  streamnum = 0;
  nstreams = 1;

  stream = init_sprng(streamnum,nstreams,SEED,SPRNG_DEFAULT); /*initialize stream */
  printf("Print information about random number stream:\n");
  print_sprng(stream);


  /*********************** print random numbers *****************************/
            
  printf("Printing 3 random numbers in [0,1):\n");
  for (i=0;i<3;i++)
  {
    rn = sprng(stream);    /* generate a double precision random number     */
    printf("%f\n", rn);
  }

  /**************************** free memory *********************************/
            
  free_sprng(stream);     /* free memory used to store stream state         */

  /********************** Try using freed stream ****************************/
            
  fprintf(stderr,"Expect a SPRNG error message on the use of an invalid stream ID\n");
  
  rn = sprng(stream);
  printf("sprng returns %f on being given an invalid stream ID\n", rn);
}
