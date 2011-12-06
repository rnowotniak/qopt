/* Function definitions of utility functions */

# include <stdio.h>
# include <stdlib.h>
# include <math.h>

# include "global.h"
# include "sub.h"
# include "rand.h"

/* Code to allocate memory to global variables being used in evaluation of functions */
void allocate_memory ()
{
    int i, j, k;
    norm_x = (long double *)malloc(nreal*sizeof(long double));
    norm_f = (long double *)malloc(nfunc*sizeof(long double));
    trans_x = (long double *)malloc(nreal*sizeof(long double));
    basic_f = (long double *)malloc(nfunc*sizeof(long double));
    temp_x1 = (long double *)malloc(nreal*sizeof(long double));
    temp_x2 = (long double *)malloc(nreal*sizeof(long double));
    temp_x3 = (long double *)malloc(nreal*sizeof(long double));
    temp_x4 = (long double *)malloc(nreal*sizeof(long double));
    weight = (long double *)malloc(nfunc*sizeof(long double));
    sigma = (long double *)malloc(nfunc*sizeof(long double));
    lambda = (long double *)malloc(nfunc*sizeof(long double));
    bias = (long double *)malloc(nfunc*sizeof(long double));
    o = (long double **)malloc(nfunc*sizeof(long double));
    l = (long double ***)malloc(nfunc*sizeof(long double));
    g = (long double **)malloc(nreal*sizeof(long double));
    for (i=0; i<nfunc; i++)
    {
        o[i] = (long double *)malloc(nreal*sizeof(long double));
        l[i] = (long double **)malloc(nreal*sizeof(long double));
    }
    for (i=0; i<nreal; i++)
    {
        g[i] = (long double *)malloc(nreal*sizeof(long double));
    }
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            l[i][j] = (long double *)malloc(nreal*sizeof(long double));
        }
    }
    /* Do some trivial (common) initialization here itself */
	C = 2000.0;
    for (i=0; i<nreal; i++)
    {
        norm_x[i] = 5.0;
        trans_x[i] = 0.0;
        temp_x1[i] = 0.0;
        temp_x2[i] = 0.0;
        temp_x3[i] = 0.0;
        temp_x4[i] = 0.0;
        for (j=0; j<nreal; j++)
        {
            if (i==j)
            {
                g[i][j]=1.0;
            }
            else
            {
                g[i][j]=0.0;
            }
        }
    }
    for (i=0; i<nfunc; i++)
    {
        basic_f[i] = 0.0;
        norm_f[i] = 0.0;
        weight[i] = 1.0/(long double)nfunc;
        sigma[i] = 1.0;
        lambda[i] = 1.0;
        bias[i] = 100.0*(long double)i;
        for (j=0; j<nreal; j++)
        {
            o[i][j] = 0.0;
            for (k=0; k<nreal; k++)
            {
                if (j==k)
                {
                    l[i][j][k] = 1.0;
                }
                else
                {
                    l[i][j][k] = 0.0;
                }
            }
        }
    }
    return;
}

/* Code to transform a variable vector based on function index 'count' */
void transform (long double *x, int count)
{
    int i, j;
    for (i=0; i<nreal; i++)
    {
        temp_x1[i] = x[i] - o[count][i];
    }
    for (i=0; i<nreal; i++)
    {
        temp_x2[i] = temp_x1[i]/lambda[count];
    }
    for (j=0; j<nreal; j++)
    {
        temp_x3[j] = 0.0;
        for (i=0; i<nreal; i++)
        {
            temp_x3[j] += g[i][j]*temp_x2[i];
        }
    }
    for (j=0; j<nreal; j++)
    {
        trans_x[j] = 0.0;
        for (i=0; i<nreal; i++)
        {
            trans_x[j] += l[count][i][j]*temp_x3[i];
        }
    }
    return;
}

/* Code to transform a vector (with elements 5.0) based on function index 'count' */
void transform_norm (int count)
{
    int i, j;
    for (i=0; i<nreal; i++)
    {
        temp_x2[i] = 5.0/lambda[count];
    }
    for (j=0; j<nreal; j++)
    {
        temp_x3[j] = 0.0;
        for (i=0; i<nreal; i++)
        {
            temp_x3[j] += g[i][j]*temp_x2[i];
        }
    }
    for (j=0; j<nreal; j++)
    {
        trans_x[j] = 0.0;
        for (i=0; i<nreal; i++)
        {
            trans_x[j] += l[count][i][j]*temp_x3[i];
        }
    }
    return;
}

/* Code to compute the weights for a variable vector */
void calc_weight (long double *x)
{
    int i, j;
    long double sum;
    long double max;
    max = -INF;
    for (i=0; i<nfunc; i++)
    {
        sum = 0.0;
        for (j=0; j<nreal; j++)
        {
            sum += (x[j]-o[i][j])*(x[j]-o[i][j]);
        }
        weight[i] = exp(-(sum)/(2.0*nreal*sigma[i]*sigma[i]));
        max = maximum(max,weight[i]);
    }
    sum = 0.0;
    for (i=0; i<nfunc; i++)
    {
        if (weight[i]!=max)
        {
            weight[i] *= (1.0 - pow(max,10.0));
        }
        sum += weight[i];
    }
    if (sum==0.0)
    {
        for (i=0; i<nfunc; i++)
        {
            weight[i] = 1.0/(long double)nfunc;
        }
    }
    else
    {
        for (i=0; i<nfunc; i++)
        {
            weight[i] /= sum;
        }
    }
    return;
}

/* Code to free the allocated memory */
void free_memory()
{
    int i, j;
    free (norm_x);
    free (norm_f);
    free (trans_x);
    free (basic_f);
    free (temp_x1);
    free (temp_x2);
    free (temp_x3);
    free (temp_x4);
    free (weight);
    free (sigma);
    free (lambda);
    free (bias);
    for (i=0; i<nfunc; i++)
    {
        for (j=0; j<nreal; j++)
        {
            free (l[i][j]);
        }
    }
    for (i=0; i<nfunc; i++)
    {
        free (o[i]);
        free (l[i]);
    }
    for (i=0; i<nreal; i++)
    {
        free (g[i]);
    }
    free (o);
    free (l);
    free (g);
    return;
}
