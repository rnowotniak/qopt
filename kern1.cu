__constant__ int nreal;
__constant__ int nfunc;

__device__ double ROB = 66;
__device__ double C;
__device__ double global_bias;
__device__ double *trans_x;
__device__ double *temp_x1;
__device__ double *temp_x2;
__device__ double *temp_x3;
__device__ double *temp_x4;
__device__ double *norm_x;
__device__ double *basic_f; // nfunc
__device__ double *weight; // nfunc
__device__ double *sigma; // nfunc
__device__ double *lambda; // nfunc
__device__ double *bias; // nfunc
__device__ double *norm_f; // nfunc
__device__ double **o;
__device__ double **g;
__device__ double ***l;

__device__ double *l_flat;

__device__ double calc_sphere (double *x)
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

__device__ void transform (double  * x, int count)
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
            // trans_x[j] += l[count][i][j]*temp_x3[i];
            trans_x[j] += l_flat[count * (nreal * nreal) + i * nreal + j] *temp_x3[i];
        }
    }
    return;
}

__global__ void calc_benchmark_func(double *x, double *result)
{
    transform (x, 0);
    basic_f[0] = calc_sphere (trans_x);

    //result[0] = basic_f[0];

    result[0] = basic_f[0] + bias[0];

//
//    results[threadIdx.x] = res;
}

__global__ void f(double *arg, double *result)
{
    result[threadIdx.x] = calc_sphere(arg);
    //result[threadIdx.x] = 1;
}

__global__ void test(double *result, double *o_out, double *g_out, double *l_out) {
    //MEM = *foo;
    //result[0] = sizeof(MEM_t);
    result[0] = l_flat[0];
    result[1] = C;
    result[2] = trans_x[0];
    result[3] = temp_x4[1];
    result[4] = norm_x[0];
    result[5] = norm_f[0];
    result[6] = o[1][0];
    result[7] = bias[0];

    for (int i = 0; i < nfunc; i++) {
        for (int j = 0; j < nreal; j++) {
            o_out[i * nreal + j] = o[i][j];
        }
    }

    for (int i = 0; i < nreal; i++) {
        for (int j = 0; j < nreal; j++) {
            g_out[i * nreal + j] = g[i][j];
        }
    }

    l_out[0] = 13;

    for (int i = 0; i < nfunc * nreal * nreal; i++) {
        l_out[i] = l_flat[i];
    }
}
