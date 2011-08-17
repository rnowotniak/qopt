
#define X 17
#define Y X

__constant__ int a = 3;

//__device__ double bla[a];

__global__ void test(double *m1, double *m2, double *m3, double *m4) {

    m1[0] = Y;

    return;

    m1[blockIdx.x] = blockIdx.x;
    m2[3] = 66;
    m1[13] = 15;
    m4[3] = 3.14;

}

