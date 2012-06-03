/***************************************************************************/
/*            ____Demonstrates the use of sprng and isprng____             */
/* A random number stream is initialized and a few random double precision */
/* numbers and a few integers are printed.                                 */
/***************************************************************************/

#include <stdio.h>
 
#define SIMPLE_SPRNG		/* simple interface                        */
#include "sprng.h"              /* SPRNG header file                       */

#define SEED 985456376



main()
{
  int seed, i, irn;
  double rn;


  /************************** Initialization *******************************/

  init_sprng(SEED,SPRNG_DEFAULT);     /* initialize stream                       */
  printf(" Print information about new stream:\n");
  print_sprng();	

  /*********************** print random numbers ****************************/
            
  printf(" Printing 3 random numbers in [0,1):\n");
  for (i=0;i<3;i++)
  {
    rn = sprng();		/* generate double precision random number */
    printf("%f\n",rn);
  }

  /*********************** print random integers ***************************/
            
  printf(" Printing 3 random integers in [0,2^31):\n");
  for (i=0;i<3;i++)
  {
    irn = isprng();	       /* generate an integer random number        */
    printf("%16d\n",irn);
  }

}
