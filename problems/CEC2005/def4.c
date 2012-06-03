/* Source file for various benchmark functions */
/* Hard-coded for every function. */
/* Some redundancy is present here and there */

# include <stdio.h>
# include <stdlib.h>
# include <math.h>

# include "global.h"
# include "sub.h"
# include "rand.h"

# ifdef f1
long double calc_benchmark_func(long double *x)
{
    long double res;
    transform (x, 0);
    basic_f[0] = calc_sphere (trans_x);
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f2
long double calc_benchmark_func(long double *x)
{
    long double res;
    transform (x, 0);
    basic_f[0] = calc_schwefel (trans_x);
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f3
long double calc_benchmark_func(long double *x)
{
    int i;
    long double res;
    transform (x, 0);
    basic_f[0] = 0.0;
    for (i=0; i<nreal; i++)
    {
        basic_f[0] += trans_x[i]*trans_x[i]*pow(1.0e6,i/(nreal-1.0));
    }
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f4
long double calc_benchmark_func(long double *x)
{
    long double res;
    transform (x, 0);
    basic_f[0] = calc_schwefel(trans_x)*(1.0 + 0.4*fabs(randomnormaldeviate()));
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f5
long double calc_benchmark_func(long double *x)
{
    int i, j;
    long double res;
    basic_f[0] = -INF;
    for (i=0; i<nreal; i++)
    {
        res=0.0;
        for (j=0; j<nreal; j++)
        {
            res += A[i][j]*x[j];
        }
        res = fabs(res-B[i]);
        if (basic_f[0] < res)
        {
            basic_f[0] = res;
        }
    }
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f6
long double calc_benchmark_func(long double *x)
{
    long double res;
    transform (x, 0);
    basic_f[0] = calc_rosenbrock(trans_x);
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f7
long double calc_benchmark_func(long double *x)
{
    long double res;
    transform (x, 0);
    basic_f[0] = calc_griewank(trans_x);
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f8
long double calc_benchmark_func(long double *x)
{
    long double res;
    transform (x, 0);
    basic_f[0] = calc_ackley(trans_x);
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f9
long double calc_benchmark_func(long double *x)
{
    long double res;
    transform (x, 0);
    basic_f[0] = calc_rastrigin(trans_x);
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f10
long double calc_benchmark_func(long double *x)
{
    long double res;
    transform (x, 0);
    basic_f[0] = calc_rastrigin(trans_x);
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f11
long double calc_benchmark_func(long double *x)
{
    int i;
    long double res;
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform (x, 0);
    basic_f[0] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f12
long double calc_benchmark_func(long double *x)
{
    long double res;
    long double sum1, sum2;
    int i, j;
    basic_f[0] = 0.0;
    for (i=0; i<nreal; i++)
    {
        sum1 = 0.0;
        sum2 = 0.0;
        for (j=0; j<nreal; j++)
        {
            sum1 += A[i][j]*sin(alpha[j]) + B[i][j]*cos(alpha[j]);
            sum2 += A[i][j]*sin(x[j]) + B[i][j]*cos(x[j]);
        }
        basic_f[0] += pow((sum1-sum2),2.0);
    }
    res = basic_f[0] + bias[0];
    return (res);
}
# endif

# ifdef f13
long double calc_benchmark_func(long double *x)
{
    int i;
    long double temp;
    long double res;
    transform(x,0);
    res = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        res += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    res += (temp*temp)/4000.0 - cos(temp) + 1.0 + bias[0];
    return (res);
}
# endif

# ifdef f14
long double calc_benchmark_func(long double *x)
{
    int i;
    long double temp1, temp2;
    long double res;
    transform(x,0);
    res = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        res += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    res += 0.5 + (temp1-0.5)/(pow(temp2,2.0)) + bias[0];
    return (res);
}
# endif

# ifdef f15
void calc_benchmark_norm()
{
    int i;
    transform_norm (0);    norm_f[0] = calc_rastrigin(trans_x);
    transform_norm (1);    norm_f[1] = calc_rastrigin(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform_norm (2);    norm_f[2] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (3);    norm_f[3] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (4);    norm_f[4] = calc_griewank(trans_x);
    transform_norm (5);    norm_f[5] = calc_griewank(trans_x);
    transform_norm (6);    norm_f[6] = calc_ackley(trans_x);
    transform_norm (7);    norm_f[7] = calc_ackley(trans_x);
    transform_norm (8);    norm_f[8] = calc_sphere(trans_x);
    transform_norm (9);    norm_f[9] = calc_sphere(trans_x);
    return;
}

long double calc_benchmark_func(long double *x)
{
    int i;
    long double res;
    transform (x, 0);    basic_f[0] = calc_rastrigin(trans_x);
    transform (x, 1);    basic_f[1] = calc_rastrigin(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform (x, 2);    basic_f[2] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 3);    basic_f[3] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 4);    basic_f[4] = calc_griewank(trans_x);
    transform (x, 5);    basic_f[5] = calc_griewank(trans_x);
    transform (x, 6);    basic_f[6] = calc_ackley(trans_x);
    transform (x, 7);    basic_f[7] = calc_ackley(trans_x);
    transform (x, 8);    basic_f[8] = calc_sphere(trans_x);
    transform (x, 9);    basic_f[9] = calc_sphere(trans_x);
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] *= C/norm_f[i];
    }
    calc_weight(x);
    res = global_bias;
    for (i=0; i<nfunc; i++)
    {
        res += weight[i]*(basic_f[i]+bias[i]);
    }
    return (res);
}
# endif

# ifdef f16
void calc_benchmark_norm()
{
    int i;
    transform_norm (0);    norm_f[0] = calc_rastrigin(trans_x);
    transform_norm (1);    norm_f[1] = calc_rastrigin(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform_norm (2);    norm_f[2] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (3);    norm_f[3] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (4);    norm_f[4] = calc_griewank(trans_x);
    transform_norm (5);    norm_f[5] = calc_griewank(trans_x);
    transform_norm (6);    norm_f[6] = calc_ackley(trans_x);
    transform_norm (7);    norm_f[7] = calc_ackley(trans_x);
    transform_norm (8);    norm_f[8] = calc_sphere(trans_x);
    transform_norm (9);    norm_f[9] = calc_sphere(trans_x);
    return;
}

long double calc_benchmark_func(long double *x)
{
    int i;
    long double res;
    transform (x, 0);    basic_f[0] = calc_rastrigin(trans_x);
    transform (x, 1);    basic_f[1] = calc_rastrigin(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform (x, 2);    basic_f[2] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 3);    basic_f[3] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 4);    basic_f[4] = calc_griewank(trans_x);
    transform (x, 5);    basic_f[5] = calc_griewank(trans_x);
    transform (x, 6);    basic_f[6] = calc_ackley(trans_x);
    transform (x, 7);    basic_f[7] = calc_ackley(trans_x);
    transform (x, 8);    basic_f[8] = calc_sphere(trans_x);
    transform (x, 9);    basic_f[9] = calc_sphere(trans_x);
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] *= C/norm_f[i];
    }
    calc_weight(x);
    res = global_bias;
    for (i=0; i<nfunc; i++)
    {
        res += weight[i]*(basic_f[i]+bias[i]);
    }
    return (res);
}
# endif

# ifdef f17
void calc_benchmark_norm()
{
    int i;
    transform_norm (0);    norm_f[0] = calc_rastrigin(trans_x);
    transform_norm (1);    norm_f[1] = calc_rastrigin(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform_norm (2);    norm_f[2] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (3);    norm_f[3] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (4);    norm_f[4] = calc_griewank(trans_x);
    transform_norm (5);    norm_f[5] = calc_griewank(trans_x);
    transform_norm (6);    norm_f[6] = calc_ackley(trans_x);
    transform_norm (7);    norm_f[7] = calc_ackley(trans_x);
    transform_norm (8);    norm_f[8] = calc_sphere(trans_x);
    transform_norm (9);    norm_f[9] = calc_sphere(trans_x);
    return;
}

long double calc_benchmark_func(long double *x)
{
    int i;
    long double res;
    transform (x, 0);    basic_f[0] = calc_rastrigin(trans_x);
    transform (x, 1);    basic_f[1] = calc_rastrigin(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform (x, 2);    basic_f[2] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 3);    basic_f[3] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 4);    basic_f[4] = calc_griewank(trans_x);
    transform (x, 5);    basic_f[5] = calc_griewank(trans_x);
    transform (x, 6);    basic_f[6] = calc_ackley(trans_x);
    transform (x, 7);    basic_f[7] = calc_ackley(trans_x);
    transform (x, 8);    basic_f[8] = calc_sphere(trans_x);
    transform (x, 9);    basic_f[9] = calc_sphere(trans_x);
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] *= C/norm_f[i];
    }
    calc_weight(x);
    res = 0.0;
    for (i=0; i<nfunc; i++)
    {
        res += weight[i]*(basic_f[i]+bias[i]);
    }
    res = res*(1.0 + 0.2*fabs(randomnormaldeviate())) + global_bias;
    return (res);
}
# endif

# ifdef f18
void calc_benchmark_norm()
{
    int i;
    transform_norm (0);    norm_f[0] = calc_ackley(trans_x);
    transform_norm (1);    norm_f[1] = calc_ackley(trans_x);
    transform_norm (2);    norm_f[2] = calc_rastrigin(trans_x);
    transform_norm (3);    norm_f[3] = calc_rastrigin(trans_x);
    transform_norm (4);    norm_f[4] = calc_sphere(trans_x);
    transform_norm (5);    norm_f[5] = calc_sphere(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform_norm (6);    norm_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (7);    norm_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (8);    norm_f[8] = calc_griewank(trans_x);
    transform_norm (9);    norm_f[9] = calc_griewank(trans_x);
    return;
}

long double calc_benchmark_func(long double *x)
{
    int i;
    long double res;
    transform (x, 0);    basic_f[0] = calc_ackley(trans_x);
    transform (x, 1);    basic_f[1] = calc_ackley(trans_x);
    transform (x, 2);    basic_f[2] = calc_rastrigin(trans_x);
    transform (x, 3);    basic_f[3] = calc_rastrigin(trans_x);
    transform (x, 4);    basic_f[4] = calc_sphere(trans_x);
    transform (x, 5);    basic_f[5] = calc_sphere(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform (x, 6);    basic_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 7);    basic_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 8);    basic_f[8] = calc_griewank(trans_x);
    transform (x, 9);    basic_f[9] = calc_griewank(trans_x);
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] *= C/norm_f[i];
    }
    calc_weight(x);
    res = global_bias;
    for (i=0; i<nfunc; i++)
    {
        res += weight[i]*(basic_f[i]+bias[i]);
    }
    return (res);
}
# endif

# ifdef f19
void calc_benchmark_norm()
{
    int i;
    transform_norm (0);    norm_f[0] = calc_ackley(trans_x);
    transform_norm (1);    norm_f[1] = calc_ackley(trans_x);
    transform_norm (2);    norm_f[2] = calc_rastrigin(trans_x);
    transform_norm (3);    norm_f[3] = calc_rastrigin(trans_x);
    transform_norm (4);    norm_f[4] = calc_sphere(trans_x);
    transform_norm (5);    norm_f[5] = calc_sphere(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform_norm (6);    norm_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (7);    norm_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (8);    norm_f[8] = calc_griewank(trans_x);
    transform_norm (9);    norm_f[9] = calc_griewank(trans_x);
    return;
}

long double calc_benchmark_func(long double *x)
{
    int i;
    long double res;
    transform (x, 0);    basic_f[0] = calc_ackley(trans_x);
    transform (x, 1);    basic_f[1] = calc_ackley(trans_x);
    transform (x, 2);    basic_f[2] = calc_rastrigin(trans_x);
    transform (x, 3);    basic_f[3] = calc_rastrigin(trans_x);
    transform (x, 4);    basic_f[4] = calc_sphere(trans_x);
    transform (x, 5);    basic_f[5] = calc_sphere(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform (x, 6);    basic_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 7);    basic_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 8);    basic_f[8] = calc_griewank(trans_x);
    transform (x, 9);    basic_f[9] = calc_griewank(trans_x);
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] *= C/norm_f[i];
    }
    calc_weight(x);
    res = global_bias;
    for (i=0; i<nfunc; i++)
    {
        res += weight[i]*(basic_f[i]+bias[i]);
    }
    return (res);
}
# endif

# ifdef f20
void calc_benchmark_norm()
{
    int i;
    transform_norm (0);    norm_f[0] = calc_ackley(trans_x);
    transform_norm (1);    norm_f[1] = calc_ackley(trans_x);
    transform_norm (2);    norm_f[2] = calc_rastrigin(trans_x);
    transform_norm (3);    norm_f[3] = calc_rastrigin(trans_x);
    transform_norm (4);    norm_f[4] = calc_sphere(trans_x);
    transform_norm (5);    norm_f[5] = calc_sphere(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform_norm (6);    norm_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (7);    norm_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (8);    norm_f[8] = calc_griewank(trans_x);
    transform_norm (9);    norm_f[9] = calc_griewank(trans_x);
    return;
}

long double calc_benchmark_func(long double *x)
{
    int i;
    long double res;
    transform (x, 0);    basic_f[0] = calc_ackley(trans_x);
    transform (x, 1);    basic_f[1] = calc_ackley(trans_x);
    transform (x, 2);    basic_f[2] = calc_rastrigin(trans_x);
    transform (x, 3);    basic_f[3] = calc_rastrigin(trans_x);
    transform (x, 4);    basic_f[4] = calc_sphere(trans_x);
    transform (x, 5);    basic_f[5] = calc_sphere(trans_x);
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform (x, 6);    basic_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 7);    basic_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 8);    basic_f[8] = calc_griewank(trans_x);
    transform (x, 9);    basic_f[9] = calc_griewank(trans_x);
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] *= C/norm_f[i];
    }
    calc_weight(x);
    res = global_bias;
    for (i=0; i<nfunc; i++)
    {
        res += weight[i]*(basic_f[i]+bias[i]);
    }
    return (res);
}
# endif

# ifdef f21
void calc_benchmark_norm()
{
    int i;
    long double temp1, temp2, temp;
    transform_norm (0);
    norm_f[0] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        norm_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    norm_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform_norm (1);
    norm_f[1] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        norm_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    norm_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform_norm (2);    norm_f[2] = calc_rastrigin(trans_x);
    transform_norm (3);    norm_f[3] = calc_rastrigin(trans_x);
    transform_norm(4);
    norm_f[4] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        norm_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    norm_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    transform_norm(5);
    norm_f[5] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        norm_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    norm_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform_norm (6);    norm_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (7);    norm_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (8);    norm_f[8] = calc_griewank(trans_x);
    transform_norm (9);    norm_f[9] = calc_griewank(trans_x);
    return;
}

long double calc_benchmark_func(long double *x)
{
    int i;
    long double temp1, temp2, temp;
    long double res;
    transform (x, 0);
    basic_f[0] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        basic_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    basic_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform (x, 1);
    basic_f[1] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        basic_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    basic_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform (x, 2);    basic_f[2] = calc_rastrigin(trans_x);
    transform (x, 3);    basic_f[3] = calc_rastrigin(trans_x);
    transform (x, 4);
    basic_f[4] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        basic_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    basic_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    transform(x, 5);
    basic_f[5] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        basic_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    basic_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform (x, 6);    basic_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 7);    basic_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 8);    basic_f[8] = calc_griewank(trans_x);
    transform (x, 9);    basic_f[9] = calc_griewank(trans_x);
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] *= C/norm_f[i];
    }
    calc_weight(x);
    res = global_bias;
    for (i=0; i<nfunc; i++)
    {
        res += weight[i]*(basic_f[i]+bias[i]);
    }
    return (res);
}
# endif

# ifdef f22
void calc_benchmark_norm()
{
    int i;
    long double temp1, temp2, temp;
    transform_norm (0);
    norm_f[0] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        norm_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    norm_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform_norm (1);
    norm_f[1] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        norm_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    norm_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform_norm (2);    norm_f[2] = calc_rastrigin(trans_x);
    transform_norm (3);    norm_f[3] = calc_rastrigin(trans_x);
    transform_norm(4);
    norm_f[4] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        norm_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    norm_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    transform_norm(5);
    norm_f[5] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        norm_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    norm_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform_norm (6);    norm_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (7);    norm_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (8);    norm_f[8] = calc_griewank(trans_x);
    transform_norm (9);    norm_f[9] = calc_griewank(trans_x);
    return;
}

long double calc_benchmark_func(long double *x)
{
    int i;
    long double temp1, temp2, temp;
    long double res;
    transform (x, 0);
    basic_f[0] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        basic_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    basic_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform (x, 1);
    basic_f[1] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        basic_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    basic_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform (x, 2);    basic_f[2] = calc_rastrigin(trans_x);
    transform (x, 3);    basic_f[3] = calc_rastrigin(trans_x);
    transform (x, 4);
    basic_f[4] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        basic_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    basic_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    transform(x, 5);
    basic_f[5] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        basic_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    basic_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform (x, 6);    basic_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 7);    basic_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (x, 8);    basic_f[8] = calc_griewank(trans_x);
    transform (x, 9);    basic_f[9] = calc_griewank(trans_x);
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] *= C/norm_f[i];
    }
    calc_weight(x);
    res = global_bias;
    for (i=0; i<nfunc; i++)
    {
        res += weight[i]*(basic_f[i]+bias[i]);
    }
    return (res);
}
# endif

# ifdef f23
void calc_benchmark_norm()
{
    int i;
    long double temp1, temp2, temp;
    transform_norm (0);
    norm_f[0] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        norm_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    norm_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform_norm (1);
    norm_f[1] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        norm_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    norm_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform_norm (2);    norm_f[2] = calc_rastrigin(trans_x);
    transform_norm (3);    norm_f[3] = calc_rastrigin(trans_x);
    transform_norm(4);
    norm_f[4] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        norm_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    norm_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    transform_norm(5);
    norm_f[5] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        norm_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    norm_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform_norm (6);    norm_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (7);    norm_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (8);    norm_f[8] = calc_griewank(trans_x);
    transform_norm (9);    norm_f[9] = calc_griewank(trans_x);
    return;
}

long double calc_benchmark_func(long double *x)
{
    int i;
    long double temp1, temp2, temp;
    long double res;
    int a;
    long double b;
    for (i=0; i<nreal; i++)
    {
        if (fabs(x[i]-o[0][i]) >= 0.5)
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
    transform (temp_x4, 0);
    basic_f[0] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        basic_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    basic_f[0] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform (temp_x4, 1);
    basic_f[1] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        basic_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    basic_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform (temp_x4, 2);    basic_f[2] = calc_rastrigin(trans_x);
    transform (temp_x4, 3);    basic_f[3] = calc_rastrigin(trans_x);
    transform (temp_x4, 4);
    basic_f[4] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        basic_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    basic_f[4] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    transform(temp_x4, 5);
    basic_f[5] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        basic_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    basic_f[5] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform (temp_x4, 6);    basic_f[6] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (temp_x4, 7);    basic_f[7] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform (temp_x4, 8);    basic_f[8] = calc_griewank(trans_x);
    transform (temp_x4, 9);    basic_f[9] = calc_griewank(trans_x);
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] *= C/norm_f[i];
    }
    calc_weight(temp_x4);
    res = global_bias;
    for (i=0; i<nfunc; i++)
    {
        res += weight[i]*(basic_f[i]+bias[i]);
    }
    return (res);
}
# endif

# if defined f24 || defined f25
void calc_benchmark_norm()
{
    int i;
    long double temp1, temp2, temp;
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    transform_norm (0);    norm_f[0] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);
    transform_norm (1);
    norm_f[1] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        norm_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    norm_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    transform_norm (2);
    norm_f[2] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        norm_f[2] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    norm_f[2] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    transform_norm (3);    norm_f[3] = calc_ackley(trans_x);
    transform_norm (4);    norm_f[4] = calc_rastrigin(trans_x);
    transform_norm (5);    norm_f[5] = calc_griewank(trans_x);
    transform_norm (6);
    norm_f[6] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        norm_f[6] += nc_schaffer(trans_x[i], trans_x[i+1]);
    }
    norm_f[6] += nc_schaffer(trans_x[nreal-1], trans_x[0]);
    transform_norm(7);    norm_f[7] = nc_rastrigin(trans_x);
    transform_norm (8);
    norm_f[8] = 0.0;
    for (i=0; i<nreal; i++)
    {
        norm_f[8] += trans_x[i]*trans_x[i]*pow(1.0e6,i/(nreal-1.0));
    }
    transform_norm (9);    norm_f[9] = calc_sphere(trans_x)*(1.0 + 0.1*fabs(randomnormaldeviate()));
    return;
}

long double calc_benchmark_func(long double *x)
{
    int i;
    long double temp1, temp2, temp;
    long double res;
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 0.0;
    }
    /* First function */
    transform (x, 0);    basic_f[0] = calc_weierstrass(trans_x) - calc_weierstrass(norm_x);

    /* Second function */
    transform (x, 1);
    basic_f[1] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp1 = pow((sin(sqrt(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0)))),2.0);
        temp2 = 1.0 + 0.001*(pow(trans_x[i],2.0)+pow(trans_x[i+1],2.0));
        basic_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));
    }
    temp1 = pow((sin(sqrt(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0)))),2.0);
    temp2 = 1.0 + 0.001*(pow(trans_x[nreal-1],2.0)+pow(trans_x[0],2.0));
    basic_f[1] += 0.5 + (temp1-0.5)/(pow(temp2,2.0));

    /* Third Function */
    transform (x, 2);
    basic_f[2] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        temp = 100.0*pow((trans_x[i]*trans_x[i]-trans_x[i+1]),2.0) + 1.0*pow((trans_x[i]-1.0),2.0);
        basic_f[2] += (temp*temp)/4000.0 - cos(temp) + 1.0;
    }
    temp = 100.0*pow((trans_x[nreal-1]*trans_x[nreal-1]-trans_x[0]),2.0) + 1.0*pow((trans_x[nreal-1]-1.0),2.0);
    basic_f[2] += (temp*temp)/4000.0 - cos(temp) + 1.0;

    transform (x, 3);    basic_f[3] = calc_ackley(trans_x);
    transform (x, 4);    basic_f[4] = calc_rastrigin(trans_x);
    transform (x, 5);    basic_f[5] = calc_griewank(trans_x);

    /* Seventh Function */
    transform (x, 6);
    basic_f[6] = 0.0;
    for (i=0; i<nreal-1; i++)
    {
        basic_f[6] += nc_schaffer(trans_x[i], trans_x[i+1]);
    }
    basic_f[6] += nc_schaffer(trans_x[nreal-1], trans_x[0]);

    transform (x, 7);    basic_f[7] = nc_rastrigin(trans_x);

    transform (x, 8);
    basic_f[8] = 0.0;
    for (i=0; i<nreal; i++)
    {
        basic_f[8] += trans_x[i]*trans_x[i]*pow(1.0e6,i/(nreal-1.0));
    }
    transform (x, 9);    basic_f[9] = (calc_sphere(trans_x))*(1.0 + 0.1*fabs(randomnormaldeviate()));
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] *= C/norm_f[i];
    }
    calc_weight(x);
    res = global_bias;
    for (i=0; i<nfunc; i++)
    {
        res += weight[i]*(basic_f[i]+bias[i]);
    }
    return (res);
}
# endif
