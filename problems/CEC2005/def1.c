/* Function definitions of basic functions */

# include <stdio.h>
# include <stdlib.h>
# include <math.h>

# include "global.h"
# include "sub.h"
# include "rand.h"

/* Code to evaluate ackley's function */
long double calc_ackley (long double *x)
{
    int i;
    long double sum1, sum2, res;
    sum1 = 0.0;
    sum2 = 0.0;
    for (i=0; i<nreal; i++)
    {
        sum1 += x[i]*x[i];
        sum2 += cos(2.0*PI*x[i]);
    }
    sum1 = -0.2*sqrt(sum1/nreal);
    sum2 /= nreal;
    res = 20.0 + E - 20.0*exp(sum1) - exp(sum2);
    return (res);
}

/* Code to evaluate rastrigin's function */
long double calc_rastrigin (long double *x)
{
    int i;
    long double res;
    res = 0.0;
    for (i=0; i<nreal; i++)
    {
        res += (x[i]*x[i] - 10.0*cos(2.0*PI*x[i]) + 10.0);
    }
    return (res);
}

/* Code to evaluate weierstrass's function */
long double calc_weierstrass (long double *x)
{
    int i, j;
    long double res;
    long double sum;
    long double a, b;
    int k_max;
    a = 0.5;
    b = 3.0;
    k_max = 20;
    res = 0.0;
    for (i=0; i<nreal; i++)
    {
        sum = 0.0;
        for (j=0; j<=k_max; j++)
        {
            sum += pow(a,j)*cos(2.0*PI*pow(b,j)*(x[i]+0.5));
        }
        res += sum;
    }
    return (res);
}

/* Code to evaluate griewank's function */
long double calc_griewank (long double *x)
{
    int i;
    long double s, p;
    long double res;
    s = 0.0;
    p = 1.0;
    for (i=0; i<nreal; i++)
    {
        s += x[i]*x[i];
        p *= cos(x[i]/sqrt(1.0+i));
    }
    res = 1.0 + s/4000.0 - p;
    return (res);
}

/* code to evaluate sphere function */
long double calc_sphere (long double *x)
{
    int i;
    long double res;
    res = 0.0;
    for (i=0; i<nreal; i++)
    {
        res += x[i]*x[i];
    }
    return (res);
}

/* Code to evaluate schwefel's function */
long double calc_schwefel (long double *x)
{
    int i, j;
    long double sum1, sum2;
    sum1 = 0.0;
    for (i=0; i<nreal; i++)
    {
        sum2 = 0.0;
        for (j=0; j<=i; j++)
        {
            sum2 += x[j];
        }
        sum1 += sum2*sum2;
    }
    return (sum1);
}

/* Code to evaluate rosenbrock's function */
long double calc_rosenbrock (long double *x)
{
    int i;
    long double res;
    res = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        res += 100.0*pow((x[i]*x[i]-x[i+1]),2.0) + 1.0*pow((x[i]-1.0),2.0);
    }
    return (res);
}

/* Code to evaluate schaffer's function and rounding-off variables */
long double nc_schaffer (long double x, long double y)
{
    int i;
    int a;
    long double b;
    long double res;
    long double temp1, temp2;
    long double t1[2], t2[2];
    t1[0] = x;
    t1[1] = y;
    for (i=0; i<2; i++)
    {
        if (fabs(t1[i]) >= 0.5)
        {
            res = 2.0*t1[i];
            a = res;
            b = fabs(res-a);
            if (b<0.5)
            {
                t2[i] = a/2.0;
            }
            else
            {
                if (res<=0.0)
                {
                    t2[i] = (a-1.0)/2.0;
                }
                else
                {
                    t2[i] = (a+1.0)/2.0;
                }
            }
        }
        else
        {
            t2[i] = t1[i];
        }
    }
    temp1 = pow((sin(sqrt(pow(t2[0],2.0)+pow(t2[1],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(t2[0],2.0)+pow(t2[1],2.0));
    res = 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    return (res);
}

/* Code to evaluate rastrigin's function and rounding-off variables */
long double nc_rastrigin (long double *x)
{
    int i;
    int a;
    long double b;
    long double res;
    for (i=0; i<nreal; i++)
    {
        if (fabs(x[i]) >= 0.5)
        {
            res = 2.0*x[i];
            a = res;
            b = fabs(res-a);
            if (b<0.5)
            {
                temp_x4[i] = a/2.0;
            }
            else
            {
                if (res<=0.0)
                {
                    temp_x4[i] = (a-1.0)/2.0;
                }
                else
                {
                    temp_x4[i] = (a+1.0)/2.0;
                }
            }
        }
        else
        {
            temp_x4[i] = x[i];
        }
    }
    res = 0.0;
    for (i=0; i<nreal; i++)
    {
        res += (temp_x4[i]*temp_x4[i] - 10.0*cos(2.0*PI*temp_x4[i]) + 10.0);
    }
    return (res);
}
