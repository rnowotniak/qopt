const int nreal = 2;

__device__ double ROB = 66;
__device__ double C;

typedef struct {
    double C;
    double global_bias;
    double *trans_x;
    double *temp_x1;
    double *temp_x2;
    double *temp_x3;
    double *temp_x4;
    double *norm_x;
    double *basic_f; // nfunc
    double *weight; // nfunc
    double *sigma; // nfunc
    double *lambda; // nfunc
    double *bias; // nfunc
    double *norm_f; // nfunc
    double **o;
    double **g;
    double ***l;
    int test;
} MEM_t;

__device__ MEM_t MEM;

__device__ double calc_sphere (double *x)
{
    int i;
    double res;
    res = 0.0;
    for (i=0; i<nreal; i++)
    {
        res += x[threadIdx.x]*x[threadIdx.x];
    }
    return (res);
}

__device__ void transform (double *x, int count)
{
    int i, j;
    for (i=0; i<nreal; i++)
    {
        MEM.temp_x1[i] = x[i] - MEM.o[count][i];
    }
    for (i=0; i<nreal; i++)
    {
        MEM.temp_x2[i] = MEM.temp_x1[i]/MEM.lambda[count];
    }
    for (j=0; j<nreal; j++)
    {
        MEM.temp_x3[j] = 0.0;
        for (i=0; i<nreal; i++)
        {
            MEM.temp_x3[j] += MEM.g[i][j]*MEM.temp_x2[i];
        }
    }
    for (j=0; j<nreal; j++)
    {
        MEM.trans_x[j] = 0.0;
        for (i=0; i<nreal; i++)
        {
            MEM.trans_x[j] += MEM.l[count][i][j]*MEM.temp_x3[i];
        }
    }
    return;
}

__global__ void calc_benchmark_func(double *x, double *results)
{
    double res;
    transform (x, 0);
    MEM.basic_f[0] = calc_sphere (MEM.trans_x);
    res = MEM.basic_f[0] + MEM.bias[0];

    results[threadIdx.x] = res;
}

__global__ void f(double *arg, double *result)
{
    result[threadIdx.x] = calc_sphere(arg);
    //result[threadIdx.x] = 1;
}

__global__ void test(MEM_t *foo, double *result) {
    MEM = *foo;
    result[0] = sizeof(MEM_t);
    result[1] = MEM.C;
    result[2] = MEM.test;
    result[3] = MEM.trans_x[1];
    result[4] = MEM.sigma[0];
    result[5] = MEM.norm_f[2];
    result[6] = ROB;
}
