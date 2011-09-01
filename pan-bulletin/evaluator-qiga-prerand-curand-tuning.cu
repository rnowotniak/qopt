/*
 * Quantum-Inspired Genetic Algorithm implementation in GPU (CUDA)
 * Copyright (C) 2011   Robert Nowotniak <rnowotniak@kis.p.lodz.pl>
 *
 * References:
 *    [1] Han, K.H. and Kim, J.H. Genetic quantum algorithm and its application
 *        to combinatorial optimization problem, 2000
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cuda_runtime_api.h>
#include <curand_kernel.h>

#ifndef TARGET_FITNESS
#define TARGET_FITNESS 1450
#endif

#define EPSILON 10e-9f
#undef M_PI
#undef M_PI_2
#undef M_PI_4
#define M_PI	3.14159265358979323846f	/* pi float */
#define M_PI_2	1.57079632679489661923f	/* pi/2 float */
#define M_PI_4	0.78539816339744830962f	/* pi/4 float */

/*
 * GPU stuff
 */
#ifndef GPU_DEVICES
#define GPU_DEVICES 1
#endif

#ifndef GRIDWIDTH
#define GRIDWIDTH 50
#endif
#ifndef GRIDHEIGHT
#define GRIDHEIGHT 1
#endif

/*
 * QiGA Algorithm settings and data structures
 */
#define MAXGEN 500
const int chromlen = 250;
const int popsize = 10;
__shared__ float Q[popsize][chromlen]; // quantum population
__shared__ char P[popsize][chromlen];  // observed classical population
__shared__ float fvals[popsize];       // fitness of the individuals
__shared__ char best[chromlen];        // best chromosome in the population
__shared__ float bestval;              // fitness of the best individual
__device__ float evalsperformed[GRIDWIDTH*GRIDHEIGHT];   // number of evaluations performed


const int grid_width = GRIDWIDTH;
const int grid_height = GRIDHEIGHT;

inline void __safeCall( cudaError err, const char *file, const int line )
{
	if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error:\n%s\n", file, line, cudaGetErrorString(err));
		fflush(stderr);
		exit(-1);
	}
}
#define safeCall(err)           __safeCall      (err, __FILE__, __LINE__)

/*
 * Threads localization macros
 */
#define TID (threadIdx.x)
#define BID (blockIdx.y * gridDim.x + blockIdx.x)
#define GTID (threadIdx.x + (grid_width * chromlen * blockIdx.y) + (chromlen * blockIdx.x))

#define Qij (Q[i][j])

/*
 * The algorithm data
 */
// Lookup table: rotation angles in Qubit state spaces
__constant__ float lookup_table[2][2][2] = { // [x][b][f(x)>=f(b)]
	0,
	0,
	0,
	0.05 * M_PI,
	0.01 * M_PI,
	0.025 * M_PI,
	0.005 * M_PI,
	0.025 * M_PI
};
// 0.0 0.0 0.0 0.15707963267948966 0.031415926535897934 0.078539816339744828 0.015707963267948967 0.078539816339744828   (orig)
// 0.9 0.0 0.0 0.15707963267948966 0.031415926535897934 0.078539816339744828 0.015707963267948967 0.078539816339744828
// 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5


// Rotation directions
__constant__ float signs_table[2][2][2][4] = {// [x][b][f(x)>=f(b)][s(alpha*beta)]
	 0,  0,  0,  0,
	 0,  0,  0,  0,
	 0,  0,  0,  0,
	-1, +1, +1,  0,
	-1, +1, +1,  0,
	+1, -1,  0, +1,
	+1, -1,  0, +1,
	+1, -1,  0, +1,
};

/* Knapsack problem data {{{ */
#define items_count 250
#define CAPACITY 673.8677805
__constant__ float items[items_count][2] = { // (weight, profit) pairs
	{9.209064f, 14.209064f},
	{9.500999f, 14.500999f},
	{6.963295f, 11.963295f},
	{2.279285f, 7.279285f},
	{3.032726f, 8.032726f},
	{1.274258f, 6.274258f},
	{3.946579f, 8.946579f},
	{1.483850f, 6.483850f},
	{8.397372f, 13.397372f},
	{9.242843f, 14.242843f},
	{8.035211f, 13.035211f},
	{4.006164f, 9.006164f},
	{6.866686f, 11.866686f},
	{5.449048f, 10.449048f},
	{2.131032f, 7.131032f},
	{9.137101f, 14.137101f},
	{1.995695f, 6.995695f},
	{8.548770f, 13.548770f},
	{9.450672f, 14.450672f},
	{2.586045f, 7.586045f},
	{3.923305f, 8.923305f},
	{4.484013f, 9.484013f},
	{1.063318f, 6.063318f},
	{3.620432f, 8.620432f},
	{2.567424f, 7.567424f},
	{1.464839f, 6.464839f},
	{7.360391f, 12.360391f},
	{3.350601f, 8.350601f},
	{6.020803f, 11.020803f},
	{1.856454f, 6.856454f},
	{7.863757f, 12.863757f},
	{4.146012f, 9.146012f},
	{5.098071f, 10.098071f},
	{2.066206f, 7.066206f},
	{7.457775f, 12.457775f},
	{7.568270f, 12.568270f},
	{1.568819f, 6.568819f},
	{6.037302f, 11.037302f},
	{8.885883f, 13.885883f},
	{3.902923f, 8.902923f},
	{8.596297f, 13.596297f},
	{4.384640f, 9.384640f},
	{1.230826f, 6.230826f},
	{3.871946f, 8.871946f},
	{2.967615f, 7.967615f},
	{2.940307f, 7.940307f},
	{9.480380f, 14.480380f},
	{8.974063f, 13.974063f},
	{3.094555f, 8.094555f},
	{2.810172f, 7.810172f},
	{2.475128f, 7.475128f},
	{8.169691f, 13.169691f},
	{2.129828f, 7.129828f},
	{4.625228f, 9.625228f},
	{9.009089f, 14.009089f},
	{1.786109f, 6.786109f},
	{4.601663f, 9.601663f},
	{4.143615f, 9.143615f},
	{6.027177f, 11.027177f},
	{8.724334f, 13.724334f},
	{4.034586f, 9.034586f},
	{1.497589f, 6.497589f},
	{1.742364f, 6.742364f},
	{7.276174f, 12.276174f},
	{5.059221f, 10.059221f},
	{8.471348f, 13.471348f},
	{4.934693f, 9.934693f},
	{7.681153f, 12.681153f},
	{8.083970f, 13.083970f},
	{4.934716f, 9.934716f},
	{2.263453f, 7.263453f},
	{1.113841f, 6.113841f},
	{7.405829f, 12.405829f},
	{7.023024f, 12.023024f},
	{4.526548f, 9.526548f},
	{8.391742f, 13.391742f},
	{1.265338f, 6.265338f},
	{4.225184f, 9.225184f},
	{9.346990f, 14.346990f},
	{8.017472f, 13.017472f},
	{8.368631f, 13.368631f},
	{4.537072f, 9.537072f},
	{2.790101f, 7.790101f},
	{5.193468f, 10.193468f},
	{8.673214f, 13.673214f},
	{2.135691f, 7.135691f},
	{9.830011f, 14.830011f},
	{3.639049f, 8.639049f},
	{9.158919f, 14.158919f},
	{4.078815f, 9.078815f},
	{9.295549f, 14.295549f},
	{9.940520f, 14.940520f},
	{1.989507f, 6.989507f},
	{2.691199f, 7.691199f},
	{6.871154f, 11.871154f},
	{9.138547f, 14.138547f},
	{3.832792f, 8.832792f},
	{9.903747f, 14.903747f},
	{3.980833f, 8.980833f},
	{5.980027f, 10.980027f},
	{6.879197f, 11.879197f},
	{6.668758f, 11.668758f},
	{8.668962f, 13.668962f},
	{3.370572f, 8.370572f},
	{1.711501f, 6.711501f},
	{3.639203f, 8.639203f},
	{4.503730f, 9.503730f},
	{6.652672f, 11.652672f},
	{6.172294f, 11.172294f},
	{9.868470f, 14.868470f},
	{3.639828f, 8.639828f},
	{9.069824f, 14.069824f},
	{8.344210f, 13.344210f},
	{9.583991f, 14.583991f},
	{6.807006f, 11.807006f},
	{4.956325f, 9.956325f},
	{2.465226f, 7.465226f},
	{1.130236f, 6.130236f},
	{6.225003f, 11.225003f},
	{1.346161f, 6.346161f},
	{3.279086f, 8.279086f},
	{5.523816f, 10.523816f},
	{7.965773f, 12.965773f},
	{4.265617f, 9.265617f},
	{7.837914f, 12.837914f},
	{5.370839f, 10.370839f},
	{7.034771f, 12.034771f},
	{5.181167f, 10.181167f},
	{6.538136f, 11.538136f},
	{1.215336f, 6.215336f},
	{3.715852f, 8.715852f},
	{2.344877f, 7.344877f},
	{1.990738f, 6.990738f},
	{8.095537f, 13.095537f},
	{3.117505f, 8.117505f},
	{8.621608f, 13.621608f},
	{2.737633f, 7.737633f},
	{2.301033f, 7.301033f},
	{1.848134f, 6.848134f},
	{1.657176f, 6.657176f},
	{9.838199f, 14.838199f},
	{1.510446f, 6.510446f},
	{8.206775f, 13.206775f},
	{2.119422f, 7.119422f},
	{7.154484f, 12.154484f},
	{2.945387f, 7.945387f},
	{6.356535f, 11.356535f},
	{3.215255f, 8.215255f},
	{4.753217f, 9.753217f},
	{8.590775f, 13.590775f},
	{8.542450f, 13.542450f},
	{9.672801f, 14.672801f},
	{3.389450f, 8.389450f},
	{1.844335f, 6.844335f},
	{5.415267f, 10.415267f},
	{9.004033f, 14.004033f},
	{4.785607f, 9.785607f},
	{8.990792f, 13.990792f},
	{1.956372f, 6.956372f},
	{6.625668f, 11.625668f},
	{3.452405f, 8.452405f},
	{4.884734f, 9.884734f},
	{8.130650f, 13.130650f},
	{6.640585f, 11.640585f},
	{1.325496f, 6.325496f},
	{4.845726f, 9.845726f},
	{5.469026f, 10.469026f},
	{1.202187f, 6.202187f},
	{4.989763f, 9.989763f},
	{9.027440f, 14.027440f},
	{4.445667f, 9.445667f},
	{6.813935f, 11.813935f},
	{1.884421f, 6.884421f},
	{9.621641f, 14.621641f},
	{2.070025f, 7.070025f},
	{3.638994f, 8.638994f},
	{5.898874f, 10.898874f},
	{3.420652f, 8.420652f},
	{9.671293f, 14.671293f},
	{5.685244f, 10.685244f},
	{4.796756f, 9.796756f},
	{6.170502f, 11.170502f},
	{6.904318f, 11.904318f},
	{3.576083f, 8.576083f},
	{5.167100f, 10.167100f},
	{7.518714f, 12.518714f},
	{8.519183f, 13.519183f},
	{5.631435f, 10.631435f},
	{5.146649f, 10.146649f},
	{8.000621f, 13.000621f},
	{2.192487f, 7.192487f},
	{9.831648f, 14.831648f},
	{8.948573f, 13.948573f},
	{5.127426f, 10.127426f},
	{8.009721f, 13.009721f},
	{1.797579f, 6.797579f},
	{4.889081f, 9.889081f},
	{7.294262f, 12.294262f},
	{4.271433f, 9.271433f},
	{2.410177f, 7.410177f},
	{4.383687f, 9.383687f},
	{8.359841f, 13.359841f},
	{6.390287f, 11.390287f},
	{5.296157f, 10.296157f},
	{6.222357f, 11.222357f},
	{6.033325f, 11.033325f},
	{6.566735f, 11.566735f},
	{2.378530f, 7.378530f},
	{2.640191f, 7.640191f},
	{1.706108f, 6.706108f},
	{2.479060f, 7.479060f},
	{5.922474f, 10.922474f},
	{3.229590f, 8.229590f},
	{2.446752f, 7.446752f},
	{5.546282f, 10.546282f},
	{2.422066f, 7.422066f},
	{7.472880f, 12.472880f},
	{6.519311f, 11.519311f},
	{9.419698f, 14.419698f},
	{2.538206f, 7.538206f},
	{4.006143f, 9.006143f},
	{8.575075f, 13.575075f},
	{2.514438f, 7.514438f},
	{4.363820f, 9.363820f},
	{5.414904f, 10.414904f},
	{7.412454f, 12.412454f},
	{5.295305f, 10.295305f},
	{9.281536f, 14.281536f},
	{6.205887f, 11.205887f},
	{8.574046f, 13.574046f},
	{2.578445f, 7.578445f},
	{7.789379f, 12.789379f},
	{9.465943f, 14.465943f},
	{5.945856f, 10.945856f},
	{2.198917f, 7.198917f},
	{3.719810f, 8.719810f},
	{8.548244f, 13.548244f},
	{2.710386f, 7.710386f},
	{9.307696f, 14.307696f},
	{5.606563f, 10.606563f},
	{5.939238f, 10.939238f},
	{3.346261f, 8.346261f},
	{7.704282f, 12.704282f},
	{3.549227f, 8.549227f},
	{5.031655f, 10.031655f},
	{8.408095f, 13.408095f},
	{9.280075f, 14.280075f},
	{4.989402f, 9.989402f},
	{2.923170f, 7.923170f},
	{9.280242f, 14.280242f},
};
// repair procedure exactly from Han's paper
__device__ void repairKnapsack(char *x) {
	float weight;
	int overfilled;
	int i,j;
	weight = 0;
	for (i = 0; i < items_count; i++) {
		weight += items[i][0] * (x[i] == '1' ? 1 : 0);
	}
	overfilled = weight > CAPACITY;
	for (i = 0; i < items_count && overfilled; i++) {
		weight -= x[i] == '1' ? items[i][0] : 0;
		x[i] = '0';
		overfilled = weight > CAPACITY;
	}
	for (j = 0; j < items_count && overfilled == 0; j++) {
		weight += x[j] == '0' ? items[j][0] : 0;
		x[j] = '1';
		overfilled = weight > CAPACITY;
	}
	x[j-1] = '0';
}
__device__ float fknapsack(char *k) {
	int i;
	float price = 0; // total price of k knapsack
	for (i = 0; i < items_count; i++) {
		price += items[i][1] * (k[i] == '1' ? 1 : 0);
	}
	return price;
}
/*
__device__ void repairKnapsack(int whichKnapsack) {
//	char *x = P[whichKnapsack];
//	P[whichKnapsack][0] = 'R';
//	x[1] = 'N';
//	x[2] = x[1] + 1;
//	x[3] = x[2] + 1;
	float weight;
	int overfilled;
	int i,j;
	weight = 0;
	for (i = 0; i < items_count; i++) {
		weight += items[i][0] * (P[whichKnapsack][i] == '1' ? 1 : 0);
	}
	overfilled = weight > CAPACITY;
	for (i = 0; i < items_count && overfilled; i++) {
		weight -= P[whichKnapsack][i] == '1' ? items[i][0] : 0;
		P[whichKnapsack][i] = '0';
		overfilled = weight > CAPACITY;
	}
	for (j = 0; j < items_count && overfilled == 0; j++) {
		weight += P[whichKnapsack][j] == '0' ? items[j][0] : 0;
		P[whichKnapsack][j] = '1';
		overfilled = weight > CAPACITY;
	}
	P[whichKnapsack][j-1] = '0';
}
__device__ void fknapsack(int whichKnapsack) {
	int i;
	float price = 0; // total price of k knapsack
	for (i = 0; i < items_count; i++) {
		price += items[i][1] * (P[whichKnapsack][i] == '1' ? 1 : 0);
	}
	fvals[whichKnapsack] = price;
	//return price;
}
	*/
/* }}} KNAPSACK */


/*
 * Quantum genes initialization stage
 */
__device__ void initialize() {
	int i;
	for (i = 0; i < popsize; i++) {
		Q[i][TID] = M_PI_4; // equal superposition of states
	}
}

/*
 * Individuals evaluation stage
 */
__device__ void evaluate() {
	if (TID < popsize) {
		//fknapsack(TID);
		fvals[TID] = fknapsack(P[TID]);
	}
}

/*
 * Observation of classical population stage;
 * Sampling the search space with respect to the quantum population probability distributions
 */
__device__ void observe(curandState *rngStates, int gen) {
	curandState local_state = rngStates[GTID];
	int i;
	// XXX (10 -> popsize)
	#pragma unroll 10 
	for (i = 0; i < popsize; i++) {
		float alpha = __cosf(Q[i][TID]);
		/* XXX
		 * Sampling the search space, critical stage of the algorithm
		 */

		//int idx = (chromlen * MAXGEN) * i + gen * chromlen + GTID;
		//int idx = i * chromlen + GTID;  -- for single generation
		//int idx = (chromlen * popsize) * gen + i * chromlen + GTID; -- for MAXGEN generations
		//int idx =  (BID * chromlen * popsize * MAXGEN)+ (chromlen * popsize) * gen + i * chromlen + GTID;
		//float r = random[idx];  // curand_uniform(&rngStates[GTID]);

		float r = curand_uniform(&local_state);
		P[i][TID] = (r < alpha*alpha) ? '0' : '1';
	}
	rngStates[GTID] = local_state;
}

__device__ void storebest() {
	if (TID == 0) {
		int i;
		float val = -1;
		int bindex;
		#pragma unroll 10 
		for (i = 0; i < popsize; i++) {
			if (fvals[i] > val) {
				val = fvals[i];
				bindex = i;
			}
		}
		if (val > bestval) {
			bestval = val;
			memcpy(best, P[bindex], chromlen);
		}
	}
}

/*
 * Update stage -- quantum genetic operators; rotations in qubit state spaces
 */
__device__ void update() {
	int i;
	int fxGTfb;
	float delta;
	int x;
	int sindex;
	float sign, cangle;
	for (i = 0; i < popsize; i++) {
		fxGTfb = fvals[i] >= bestval;
		x = P[i][TID];
		delta = lookup_table[x=='1'][best[TID]=='1'][fxGTfb];
		cangle = fmodf(Q[i][TID], M_PI); // Qij angle casted into <0,M_PI> interval
		// cf. Table 1 in [1]
		sindex =
			// if alpha * beta:
			(cangle > EPSILON && cangle < M_PI_2 - EPSILON) ? 0 :
			// if alpha * beta < 0:
			(cangle < M_PI - EPSILON && cangle > M_PI_2 + EPSILON) ? 1 :
			// if alpha == 0:
			(fabsf((fmodf((M_PI_2 + Q[i][TID]),M_PI) - M_PI_2)) < EPSILON) ? 2 : // XXX it was fabsL
			// if beta == 0:
			3;
		sign = signs_table[x=='1'][best[TID]=='1'][fxGTfb][sindex];
		Q[i][TID] += sign * delta;
	}
}

__device__ void repair() {
	if (TID < popsize) {
		repairKnapsack(P[TID]);
	}
}


__global__ void initRNG(unsigned long long seed, curandState *rngStates) {
		curand_init(seed, GTID, 0, &rngStates[GTID]);
}

__global__ void qiga(char *BESTgmem, float *FITNESSgmem, curandState *rngStates) {
	//curand_init(seed, GTID, 0, &rngStates[GTID]);
	//__syncthreads();

	int t = 0;
	bestval = -1;

	evalsperformed[BID] = MAXGEN * popsize;

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
		if (bestval > TARGET_FITNESS) {
			evalsperformed[BID] = t * popsize;
			break;
		}
		__syncthreads();
	}

	__syncthreads();

	if (TID == 0) {
		FITNESSgmem[grid_width * blockIdx.y + blockIdx.x] = bestval;

		// FITNESSgmem[0] = lookup_table[0][0][0]; // XXX
		//memcpy(BESTgmem + BID * chromlen, best, chromlen);
	}
}

int main() {
	fprintf(stderr, "QiGA on GPU (CUDA)\n");
	fflush(stderr);

	struct timeval start_tm;
	gettimeofday(&start_tm, 0);

	int curdev;

	unsigned long long seed = time(0) + start_tm.tv_usec;
	//unsigned long long seed = 1298823875;  // fixed seed
	fprintf(stderr, "seed:%d\n", seed);
	srand(seed);

	//lookup_table[0][0][0] = 0.1;
	//printf("%d\n", sizeof(lookup_table));
	//exit(0);


	for (int dev = 0; dev < GPU_DEVICES; dev++) {
		if (fork() > 0) {
			continue;
		}

		seed += dev;

		safeCall(cudaSetDevice(dev));
		safeCall(cudaGetDevice(&curdev));
		fprintf(stderr, "Initializing device %d...\n", curdev);

		int POPULATIONS = grid_width * grid_height;

		// allocate host memory
		char *h_best;
		float *h_fit;
		h_best = (char*)calloc(chromlen * POPULATIONS, 1);
		h_fit = (float*)calloc(POPULATIONS, sizeof(float));

		// allocate device memory
		char *d_best;
		float *d_fit;
		curandState *rngStates;

		safeCall(cudaMalloc((void**)&d_best, chromlen * POPULATIONS));
		safeCall(cudaMalloc((void**)&d_fit, sizeof(float) * POPULATIONS));
		safeCall(cudaMalloc(&rngStates, POPULATIONS * chromlen * sizeof(curandState)));

		cudaError_t err;

		// copy from host memory
		safeCall(cudaMemcpy(d_best, h_best, chromlen * POPULATIONS, cudaMemcpyHostToDevice));
		safeCall(cudaMemcpy(d_fit, h_fit, sizeof(float) * POPULATIONS, cudaMemcpyHostToDevice));


		while (true) {
			seed = time(0) + start_tm.tv_usec + dev;

			// RNG initialization kernel
			initRNG<<<dim3(grid_width, grid_height), chromlen>>>(seed, rngStates);
			err = cudaGetLastError();
			if (cudaSuccess != err) {
				fprintf(stderr,"(%d)-> %s\n", curdev, cudaGetErrorString(err) );
			}
			safeCall(cudaThreadSynchronize());


			float data[2][2][2];
			int scanfret = scanf("%f %f %f %f %f",
					&data[0][1][1], &data[1][0][0], &data[1][0][1], &data[1][1][0], &data[1][1][1]);
			if (scanfret == EOF) {
				break;
			}
			safeCall(cudaMemcpyToSymbol(lookup_table, data, sizeof(data)));
			//float x;
			//safeCall(cudaMemcpyFromSymbol(&x, lookup_table, sizeof(float)));
			//printf("-> %f\n", x);


			// actual QIGA computational kernel execution
			gettimeofday(&start_tm, 0);
			//fprintf(stderr, "Starting qiga computational kernel (dev %d)\n", dev);
			//fflush(stderr);
			qiga<<<dim3(grid_width, grid_height),chromlen>>>(d_best, d_fit, rngStates);
			err = cudaGetLastError();
			if (cudaSuccess != err) {
				fprintf(stderr,"-> %s\n", cudaGetErrorString(err) );
			}
			safeCall(cudaThreadSynchronize());
			fflush(stderr);

			struct timeval stop_tm;
			gettimeofday(&stop_tm, 0);
			//fprintf(stderr,"dev: %d, %g seconds\n", curdev, (1e6 * (stop_tm.tv_sec - start_tm.tv_sec) + (stop_tm.tv_usec - start_tm.tv_usec))/1e6);

			float h_evalsperformed[GRIDWIDTH*GRIDHEIGHT];   // number of evaluations performed
			safeCall(cudaMemcpyFromSymbol(h_evalsperformed, evalsperformed, sizeof(evalsperformed)));

			safeCall(cudaMemcpy(h_best, d_best, chromlen * POPULATIONS, cudaMemcpyDeviceToHost));
			safeCall(cudaMemcpy(h_fit, d_fit, sizeof(float) * POPULATIONS, cudaMemcpyDeviceToHost));

			safeCall(cudaThreadSynchronize());
			float evals_avg = 0;
			for (int r = 0; r < POPULATIONS; r++) {
				evals_avg += h_evalsperformed[r];
			}
			evals_avg /= POPULATIONS;
			float fit_avg = 0;
			for (int r = 0; r < POPULATIONS; r++) {
				fit_avg += h_fit[r];
			}
			fit_avg /= POPULATIONS;
			printf("%f %f\n", evals_avg, fit_avg);
			// printf("d_fit: %f\n", h_fit[0]); // XXX

			fflush(stdout);
			fflush(stderr);

		}

		return 0;
	}

	for (int dev = 0; dev < GPU_DEVICES; dev++) {
		int status;
		wait(&status);
	}

	return 0;
}

//  vim: set ft=c foldmethod=marker:

