#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

inline void __safeCall( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error:\n%s\n", file, line, cudaGetErrorString(err));
		fflush(stderr);
		exit(-1);
	}
}
#define safeCall(err)           __safeCall      (err, __FILE__, __LINE__)



__constant__ const int nreal = 2;
__constant__ const int nfunc = 1;

double C;
double global_bias;
double *basic_f;
double *bias;

double *temp_x1; // zmienne
double *temp_x2; // zmienne
double *temp_x3; // zmienne
double *temp_x4; // zmienne
double *trans_x; // zmienne
double *weight; // zmienne

double *sigma;
double *lambda;
double *norm_x;
double *norm_f;
double **o;
double **g;
double ***l;

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
	norm_x = (double *)malloc(nreal*sizeof(double));
	norm_f = (double *)malloc(nfunc*sizeof(double));
	basic_f = (double *)malloc(nfunc*sizeof(double));

	trans_x = (double *)malloc(nreal*sizeof(double));
	temp_x1 = (double *)malloc(nreal*sizeof(double));
	temp_x2 = (double *)malloc(nreal*sizeof(double));
	temp_x3 = (double *)malloc(nreal*sizeof(double));
	temp_x4 = (double *)malloc(nreal*sizeof(double));
	weight = (double *)malloc(nfunc*sizeof(double));

	sigma = (double *)malloc(nfunc*sizeof(double));
	lambda = (double *)malloc(nfunc*sizeof(double));
	bias = (double *)malloc(nfunc*sizeof(double));
	o = (double **)malloc(nfunc*sizeof(double));
	l = (double ***)malloc(nfunc*sizeof(double));
	g = (double **)malloc(nreal*sizeof(double));
	for (i=0; i<nfunc; i++)
	{
		o[i] = (double *)malloc(nreal*sizeof(double));
		l[i] = (double **)malloc(nreal*sizeof(double));
	}
	for (i=0; i<nreal; i++)
	{
		g[i] = (double *)malloc(nreal*sizeof(double));
	}
	for (i=0; i<nfunc; i++)
	{
		for (j=0; j<nreal; j++)
		{
			l[i][j] = (double *)malloc(nreal*sizeof(double));
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
		weight[i] = 1.0/(double)nfunc;
		sigma[i] = 1.0;
		lambda[i] = 1.0;
		bias[i] = 100.0*(double)i;
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
double calc_sphere (double *x)
{
	int i;
	double res;
	res = 0.0;
	for (i=0; i<nreal; i++)
	{
		res += x[i]*x[i];
	}
	return (res);
}

__device__ void transform (double *x, int count)
{
	int i, j;
	double temp_x1[nreal *sizeof(double)];

//	for (i=0; i<nreal; i++)
//	{
//		temp_x1[i] = x[i] - o[count][i];
//	}
//	for (i=0; i<nreal; i++)
//	{
//		temp_x2[i] = temp_x1[i]/lambda[count];
//	}
//	for (j=0; j<nreal; j++)
//	{
//		temp_x3[j] = 0.0;
//		for (i=0; i<nreal; i++)
//		{
//			temp_x3[j] += g[i][j]*temp_x2[i];
//		}
//	}
//	for (j=0; j<nreal; j++)
//	{
//		trans_x[j] = 0.0;
//		for (i=0; i<nreal; i++)
//		{
//			trans_x[j] += l[count][i][j]*temp_x3[i];
//		}
//	}
	return;
}

__global__ void calc_benchmark_func(int aa, double *x, double *result)
{
	double res;
	transform (x, 0);
	//	basic_f[0] = calc_sphere (trans_x);
	//	res = basic_f[0] + bias[0];
	//	return (res);
	result[0] = aa + x[0] + x[1] + nreal;
}


int main (int argc, char**argv)
{

	// tutaj powinna byc jeszcze inicjalizacja, czyli np. wczytanie 'o' z pliku itp
	// a potem skopiowanie do device memory


	double hostX[2];
	hostX[0] = -3.931190E+01;
	hostX[1] = 5.889990E+01;
	double hostResult;

	double *x = NULL;
	double *result = NULL;
	safeCall(cudaMalloc((void**)&x, sizeof(double) * 2));
	safeCall(cudaMalloc((void**)&result, sizeof(double)));
	printf("%p\n", x);
	printf("%p\n", result);
	safeCall(cudaMemcpy(x, hostX, sizeof(double) * 2, cudaMemcpyHostToDevice));

	calc_benchmark_func<<<1,1>>>(3, x, result);

	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err)
		printf("error running kernel: %s\n", cudaGetErrorString(err) );
	safeCall(cudaThreadSynchronize());

	safeCall(cudaMemcpy(&hostResult, result, sizeof(double), cudaMemcpyDeviceToHost));

	printf("result: %f\n", hostResult);


	//	allocate_memory();
	//	initialize();
	//	printf("\n\n\tObjective value = %1.15LE\n", calc_benchmark_func(x));
}
