#include <curand_kernel.h>

/*

$ nvcc --ptxas-options=-v  --cubin -D NREAL=29 -D NFUNC=1 -D BLOCKSIZE=100 -arch sm_13 const-problem.cu
ptxas info    : Compiling entry function '_Z22calc_benchmark_func_f1PdS_' for 'sm_13'
ptxas info    : Used 4 registers, 8+16 bytes smem, 64928 bytes cmem[0], 8 bytes cmem[14]
                                                   ^^^^^
ptxas info    : Compiling entry function '_Z7initRNGj' for 'sm_13'
ptxas info    : Used 18 registers, 6440+0 bytes lmem, 4+16 bytes smem, 64928 bytes cmem[0], 40 bytes cmem[1], 8 bytes cmem[14]
$ nvcc --ptxas-options=-v  --cubin -D NREAL=29 -D NFUNC=1 -D BLOCKSIZE=100 -arch sm_20 const-problem.cu
ptxas info    : Compiling entry function '_Z22calc_benchmark_func_f1PdS_' for 'sm_20'
ptxas info    : Function properties for _Z22calc_benchmark_func_f1PdS_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 2 registers, 40 bytes cmem[0], 13728 bytes cmem[2], 16 bytes cmem[14]
                                                    ^^^^^
ptxas info    : Compiling entry function '_Z7initRNGj' for 'sm_20'
ptxas info    : Function properties for _Z7initRNGj
    6440 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, 4+0 bytes lmem, 36 bytes cmem[0], 13728 bytes cmem[2], 16 bytes cmem[14]
$ nvcc --ptxas-options=-v  --cubin -D NREAL=30 -D NFUNC=1 -D BLOCKSIZE=100 -arch sm_20 const-problem.cu
ptxas info    : Compiling entry function '_Z22calc_benchmark_func_f1PdS_' for 'sm_20'
ptxas info    : Function properties for _Z22calc_benchmark_func_f1PdS_
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 2 registers, 40 bytes cmem[0], 14680 bytes cmem[2], 16 bytes cmem[14]
ptxas info    : Compiling entry function '_Z7initRNGj' for 'sm_20'
ptxas info    : Function properties for _Z7initRNGj
    6440 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 22 registers, 4+0 bytes lmem, 36 bytes cmem[0], 14680 bytes cmem[2], 16 bytes cmem[14]
$ nvcc --ptxas-options=-v  --cubin -D NREAL=30 -D NFUNC=1 -D BLOCKSIZE=100 -arch sm_13 const-problem.cu
/tmp/tmpxft_000050b1_00000000-9_const-problem.cpp3.i(0): Error: Const space overflowed
$ _

*/


//__shared__ double buf[NREAL * BLOCKSIZE];

__device__ curandState *rngStates; // (sizeof(curandState) = 40 bytes)

__constant__ int nreal;
__constant__ int nfunc;

__constant__ double C;
__constant__ double global_bias;

__device__ double *sigma; // nfunc  // const? (4KB) XXX (move to constant)
__constant__ double lambda[NFUNC]; // nfunc  // const? (4KB)
__constant__ double bias[NFUNC]; // nfunc  // const? (4KB)
__constant__ double o[NFUNC][NREAL]; // const? (nfunc x nreal) (maks 4 KB)
__constant__ double g[NREAL][NREAL]; // const? (nreal x nreal) (20KB dla nreal 50)

__constant__ double l_flat[NFUNC*NREAL*NREAL];

#define GTID ( blockIdx.y * blockDim.x + threadIdx.x )

__global__ void initRNG(unsigned int seed) {
	curand_init(seed, GTID, 0, &rngStates[GTID]);
}


__global__ void calc_benchmark_func_f1(double *x, double *res)
{
	res[0] = 0;
}
