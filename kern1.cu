
__global__ void f(double *result)
{
    result[threadIdx.x] = 15;
}
