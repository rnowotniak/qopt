#include <curand_kernel.h>

#define CUDART_INF              __longlong_as_double(0x7ff0000000000000ULL)


extern "C" {

    __device__ curandState *rngStates; // (sizeof(curandState) = 40 bytes)

    __constant__ int nreal;
    __constant__ int nfunc;

    // wszystko ponizsze, co jest stale, chyba powinno byc w constant (wspolne dla wszystkich watkow)
    __device__ double C;  // const?
    __device__ double global_bias;  // const?

    __device__ double *sigma; // nfunc  // const? (4KB)
    __device__ double *lambda; // nfunc  // const? (4KB)
    __device__ double *bias; // nfunc  // const? (4KB)
    __device__ double **o; // const? (nfunc x nreal) (maks 4 KB)
    __device__ double **g; // const? (nreal x nreal) (20KB dla nreal 50)
    __device__ double ***l;  // const? (nfunc x nreal x nreal) (maks 200 KB)

    __device__ double *l_flat;

    // for function f5
    __device__ double *A; // 2d flatten, const?
    __device__ double *B; // const?


    // for parallel execution
#define GTID ( blockIdx.y * blockDim.x + threadIdx.x )

    __device__ double *g_trans_x; // RW (te chyba powinny byc w shared, ale osobne dla kazdego watku,
    __device__ double *g_temp_x1; // RW wiec to chyba powinno byc tablicami 2d)
    __device__ double *g_temp_x2; // RW (jednak w shared to sie nie zmiesci: 50 osobnikow * 50 dim * 8bytes = 20 KB)
    __device__ double *g_temp_x3; // RW (wychodzi na to, ze musi byc w global)
    __device__ double *g_temp_x4; // RW
    __device__ double *g_norm_x;  // RW
    __device__ double *g_basic_f; // nfunc  RW (ok 4 KB w przypadku 50 osobnikow)
    __device__ double *g_weight; // nfunc   RW
    __device__ double *g_norm_f; // nfunc  RW

#define trans_x (g_trans_x + nreal * GTID)
#define temp_x1 (g_temp_x1 + nreal * GTID)
#define temp_x2 (g_temp_x2 + nreal * GTID)
#define temp_x3 (g_temp_x3 + nreal * GTID)
#define temp_x4 (g_temp_x4 + nreal * GTID)
#define norm_x  (g_norm_x  + nreal * GTID)
#define basic_f (g_basic_f + nfunc * GTID)
#define weight  (g_weight  + nfunc * GTID)
#define norm_f  (g_norm_f  + nfunc * GTID)


    __global__ void initRNG(unsigned int seed) {
        curand_init(seed, GTID, 0, &rngStates[GTID]);
    }


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

    __device__ double calc_schwefel (double *x)
    {
        int i, j;
        double sum1, sum2;
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

    __device__ void transform (double *x, int count)
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


    // F1
    __global__ void calc_benchmark_func_f1(double *x, double *res)
    {
        transform (x + nreal * GTID, 0);
        basic_f[0] = calc_sphere (trans_x);
        res[GTID] = basic_f[0] + bias[0];
    }

    // F2
    __global__ void calc_benchmark_func_f2(double *x, double *res)
    {
        transform (x + nreal * GTID, 0);
        basic_f[0] = calc_schwefel (trans_x);
        res[GTID] = basic_f[0] + bias[0];
    }

    // F3
    __global__ void calc_benchmark_func_f3(double *x, double *res)
    {
        int i;
        transform (x + nreal * GTID, 0);
        basic_f[0] = 0.0;
        for (i=0; i<nreal; i++)
        {
            basic_f[0] += trans_x[i]*trans_x[i]*pow(1.0e6,i/(nreal-1.0));
        }
        res[GTID] = basic_f[0] + bias[0];
    }

    // F4
    __global__ void calc_benchmark_func_f4(double *x, double *res)
    {
        transform (x + nreal * GTID, 0);
        basic_f[0] = calc_schwefel(trans_x)*(1.0 + 0.4*fabs(curand_normal(&rngStates[GTID])));
        // basic_f[0] = calc_schwefel(trans_x)*(1.0 + 0.4*0); // no noise
        res[GTID] = basic_f[0] + bias[0];
    }

    // F5
    __global__ void calc_benchmark_func_f5(double *x, double *res)
    {
        int i, j;
        double *x_ = x + nreal * GTID;
        basic_f[0] = -CUDART_INF;
        for (i=0; i<nreal; i++)
        {
            res[GTID]=0.0;
            for (j=0; j<nreal; j++)
            {
                res[GTID] += A[i * nreal + j]*x_[j];
            }
            res[GTID] = fabs(res[GTID]-B[i]);
            if (basic_f[0] < res[GTID])
            {
                basic_f[0] = res[GTID];
            }
        }
        res[GTID] = basic_f[0] + bias[0];
    }

}
