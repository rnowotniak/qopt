#include <curand_kernel.h>

extern "C" {

    __device__ char str[10];

    __global__ void test(double *m1, unsigned int seed) {

        curandState state;
        curand_init(2012, threadIdx.x, 0, &state);

        m1[0] = sizeof(curandState);
        m1[1] = sizeof(int);
        m1[2] = 1 + seed;
        m1[3] = sizeof(m1);

        // m1[threadIdx.x] = curand_normal(&state);
    }

}

