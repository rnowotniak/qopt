#include <stdio.h>
#include <stdlib.h>

int nreal = 2;
int nfunc = 1;

long double C;
long double global_bias;
long double *trans_x;
long double *basic_f;
long double *bias;

long double *temp_x1;
long double *temp_x2;
long double *temp_x3;
long double *temp_x4;
long double *weight;
long double *sigma;
long double *lambda;
long double *norm_x;
long double *norm_f;
long double **o;
long double **g;
long double ***l;

void initialize()
{
	int i, j;
	FILE *fpt;
	fpt = fopen("input_data/sphere_func_data.txt","r");
	if (fpt==NULL)
	{
		fprintf(stderr,"\n Error: Cannot open input file for reading \n");
		exit(0);
	}
	for (i=0; i<nfunc; i++)
	{
		for (j=0; j<nreal; j++)
		{
			fscanf(fpt,"%Lf",&o[i][j]);
			printf("\n O[%d][%d] = %LE",i+1,j+1,o[i][j]);
		}
	}
	fclose(fpt);
	bias[0] = -450.0;
	return;
}

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

long double calc_benchmark_func(long double *x)
{
	long double res;
	transform (x, 0);
	basic_f[0] = calc_sphere (trans_x);
	res = basic_f[0] + bias[0];
	return (res);
}


int main (int argc, char**argv)
{
	long double x[2];
	x[0] = -3.931190E+01;
	x[1] = 5.889990E+01;
	allocate_memory();
	initialize();
	printf("\n\n\tObjective value = %1.15LE\n", calc_benchmark_func(x));
}
