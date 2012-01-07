#include <stdio.h>
#include <stdlib.h>

double myrandom_()		/* remove _ before C compilation */
{
  return (double) rand()/RAND_MAX;
}
