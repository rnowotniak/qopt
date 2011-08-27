#include <curand_kernel.h>

#define CUDART_INF              __longlong_as_double(0x7ff0000000000000ULL)

#ifndef NREAL
#error NREAL is not defined
#endif
#ifndef NFUNC
#error NFUNC is not defined
#endif
#ifndef BLOCKSIZE
#error BLOCKSIZE is not defined
#endif

extern "C" {

    //__shared__ double buf[NREAL * BLOCKSIZE];

    __device__ curandState *rngStates; // (sizeof(curandState) = 40 bytes)

    /*CONST*/__device__ int nreal;
    /*CONST*/__device__ int nfunc;

    /*CONST*/__device__ double C;
    /*CONST*/__device__ double global_bias;

    __device__ double *sigma; // nfunc  // const? (4KB) XXX (move to constant)
    /*CONST*/__device__ double lambda[NFUNC]; // nfunc  // const? (4KB)
    /*CONST*/__device__ double bias[NFUNC]; // nfunc  // const? (4KB)
    /*CONST*/__device__ double o[NFUNC][NREAL]; // const? (nfunc x nreal) (maks 4 KB)
    /*CONST*/__device__ double g[NREAL][NREAL]; // const? (nreal x nreal) (20KB dla nreal 50)

    /*CONST*/__device__ double l_flat[NFUNC*NREAL*NREAL];

    // for function f5
    __device__ double *A; // 2d flatten, const?
    __device__ double *B; // const?


    // for parallel execution (Global Thread Id)
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


    __device__ double calc_rosenbrock (double *x)
    {
        int i;
        double res;
        res = 0.0;
	//memcpy((buf + threadIdx.x * nreal), x, sizeof(double) * nreal);
        for (i=0; i<nreal-1; i++)
        {
            res += 100.0*pow((x[i]*x[i]-x[i+1]),2.0) + 1.0*pow((x[i]-1.0),2.0);
            //res += 100.0*pow(((buf + threadIdx.x * nreal)[i]*(buf + threadIdx.x * nreal)[i]-(buf + threadIdx.x * nreal)[i+1]),2.0) + 1.0*pow(((buf + threadIdx.x * nreal)[i]-1.0),2.0);
        }
        return (res);
    }

    __device__ void transform (double *x, int count)
    {
        int i, j;
	//memcpy((buf + threadIdx.x * nreal), x, sizeof(double) * nreal);
        for (i=0; i<nreal; i++)
        {
            temp_x1[i] = x[i] - o[count][i];
            //temp_x1[i] = (buf + threadIdx.x * nreal)[i] - o[count][i];
        }
        for (i=0; i<nreal; i++)
        {
            temp_x2[i] = temp_x1[i]/lambda[count];
        }
        for (j=0; j<nreal; j++)
        {
            temp_x3[j] = 0.0;
            //(buf + threadIdx.x * nreal)[j] = 0.0;
            for (i=0; i<nreal; i++)
            {
                temp_x3[j] += g[i][j]*temp_x2[i];
                //(buf + threadIdx.x * nreal)[j] += g[i][j]*temp_x2[i];
            }
        }
        for (j=0; j<nreal; j++)
        {
            trans_x[j] = 0.0;
            for (i=0; i<nreal; i++)
            {
                // trans_x[j] += l[count][i][j]*temp_x3[i];
                trans_x[j] += l_flat[count * (nreal * nreal) + i * nreal + j] *temp_x3[i];
                //trans_x[j] += l_flat[count * (nreal * nreal) + i * nreal + j] *(buf + threadIdx.x * nreal)[i];
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

    // F6
    __global__ void calc_benchmark_func_f6(double *x, double *res)
    {
        transform (x + nreal * GTID, 0);
        basic_f[0] = calc_rosenbrock(trans_x);
        res[GTID] = basic_f[0] + bias[0];
    }

    __global__ void test_time(double *x, double *res, int n) {
        for (int i = 0; i < n; i++) {
            transform (x + nreal * GTID, 0);
            basic_f[0] = calc_rosenbrock(trans_x);
            res[GTID] = basic_f[0] + bias[0];
        }
    }

}
