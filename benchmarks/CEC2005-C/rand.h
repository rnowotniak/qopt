/* Declaration for random number related variables and routines */

# ifndef _RAND_H
# define _RAND_H

# include "global.h"

# define SIMPLE_SPRNG
# include "sprng/include/sprng.h"
# define SEED 01234567

/* Variable declarations for the random number generator */
int rndcalcflag;
long double rndx1, rndx2;

/* Function declarations for the random number generator */
void randomize();
long double randomperc();
int rnd (int low, int high);
long double rndreal (long double low, long double high);
void initrandomnormaldeviate();
long double noise (long double mu, long double sigma);
long double randomnormaldeviate();

# endif
