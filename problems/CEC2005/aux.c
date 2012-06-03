/* Some auxillary functions (not part of any algorithm or procdure) */

# include <stdio.h>
# include <stdlib.h>
# include <math.h>

# include "global.h"
# include "sub.h"
# include "rand.h"

/* Function to return the maximum of two variables */
long double maximum (long double a, long double b)
{
    if (a>b)
    {
        return(a);
    }
    return (b);
}

/* Function to return the minimum of two variables */
long double minimum (long double a, long double b)
{
    if (a<b)
    {
        return (a);
    }
    return (b);
}

/* Function to return the modulus of a vector */
long double modulus (long double *x, int n)
{
    int i;
    long double res;
    res = 0.0;
    for (i=0; i<n; i++)
    {
        res += x[i]*x[i];
    }
    return (sqrt(res));
}

/* Function to return the dot product of two vecors */
long double dot (long double *a, long double *b, int n)
{
    int i;
    long double res;
    res = 0.0;
    for (i=0; i<n; i++)
    {
        res += a[i]*b[i];
    }
    return (res);
}

/* Function to return the mean of n variables */
long double mean (long double *x, int n)
{
    int i;
    long double res;
    res = 0.0;
    for (i=0; i<n; i++)
    {
        res += x[i];
    }
    return (res/(long double)n);
}
