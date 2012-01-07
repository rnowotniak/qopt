/* Definition of random number generation routines */
/* This file now acts as a wrapper for the sprng random number generator */
/* Web-site of SPRNG is http://sprng.cs.fsu.edu */

# include <stdio.h>
# include <stdlib.h>
# include <math.h>

# include "global.h"
# include "rand.h"

/* Get seed number for random and start it up */
void randomize()
{
    init_sprng(SEED,SPRNG_DEFAULT);
    return;
}

/* Fetch a single random number between 0.0 and 1.0 */
long double randomperc()
{
    return (sprng());
}

/* Fetch a single random integer between low and high including the bounds */
int rnd (int low, int high)
{
    int res;
    if (low >= high)
    {
        res = low;
    }
    else
    {
        res = low + (randomperc()*(high-low+1));
        if (res > high)
        {
            res = high;
        }
    }
    return (res);
}

/* Fetch a single random real number between low and high including the bounds */
long double rndreal (long double low, long double high)
{
    return (low + (high-low)*randomperc());
}

/* Initialize the randome generator for normal distribution */
void initrandomnormaldeviate()
{
    rndcalcflag = 1;
    return;
}

/* Return the noise value */
long double noise (long double mu, long double sigma)
{
    return((randomnormaldeviate()*sigma) + mu);
}

/* Compute the noise */
long double randomnormaldeviate()
{
    long double t;
    if(rndcalcflag)
    {
        rndx1 = sqrt(- 2.0*log(randomperc()));
        t = 6.2831853072*randomperc();
        rndx2 = sin(t);
        rndcalcflag = 0;
        return(rndx1*cos(t));
    }
    else
    {
        rndcalcflag = 1;
        return(rndx1*rndx2);
    }
}
