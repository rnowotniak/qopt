
__device__ void evaluate() {
    if (TID < popsize) {
        //fknapsack(TID);
        fvals[TID] = fknapsack(P[TID]);
    }
}

__global__ void rcqiea(char *BESTgmem, float *FITNESSgmem, curandState *rngStates) {

    int t = 0;
    bestval = -1;

    initialize();
    __syncthreads();
    observe(rngStates, t);
    __syncthreads();
    repair();
    __syncthreads();
    evaluate();
    __syncthreads();
    storebest();
    t = 0;
    while (t < MAXGEN) {
        t++;
        __syncthreads();
        observe(rngStates, t);
        __syncthreads();
        repair();
        __syncthreads();
        evaluate();
        __syncthreads();
        update();
        __syncthreads();
        storebest();
        __syncthreads();
    }

    __syncthreads();

    if (TID == 0) {
        FITNESSgmem[grid_width * blockIdx.y + blockIdx.x] = bestval;
    }
}

