#ifndef _QIEA2_H
#define _QIEA2_H 1

#include "framework.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

class QIEA2 {

	public:

		typedef double DTYPE;

		DTYPE *Q; // [popsize,N] -- quantum population
		DTYPE *P; // [K,N]       -- observed classical population  (denoted E in iQIEA)
		DTYPE *C; // [K,N]       -- crossed-over P

		int t;

		DTYPE (*bounds)[2];
		int chromlen; // N
		int popsize;  // |Q|
		int K; // K >= popsize

		DTYPE *fvals;
		DTYPE *fvalsC;
		DTYPE bestval;
		DTYPE *best;
		// DTYPE (*bestq)[2];

		Problem<DTYPE,DTYPE> *problem;

		QIEA2(int chromlen, int popsize, int K = 0) : popsize(popsize), chromlen(chromlen) {
			if (K == -1) {
				K = popsize;
			}
			printf("QIEA2::QIEA2 constructor %d %d %d\n", chromlen, popsize, K);
			assert(chromlen % 2 == 0);

			problem = NULL;
			bestval = std::numeric_limits<DTYPE>::max();

			Q = new DTYPE[5 * chromlen / 2 * popsize];
			P = new DTYPE[K * chromlen];
			C = new DTYPE[K * chromlen];

			fvals = new DTYPE[K];
			fvalsC = new DTYPE[K];
			bounds = new DTYPE[chromlen][2];
			best = new DTYPE[chromlen];
			// bestq = new DTYPE[chromlen][2];
		}

		~QIEA2() {
			delete [] Q;
			delete [] P;
			delete [] C;
			delete [] bounds;
			delete [] best;
			delete [] fvals;
			delete [] fvalsC;
			// delete [] bestq;
		}

		// the whole algorithm in C++
		void run();

		// elementary operations
		void initialize();
		void observe();
		void storebest();
		void evaluate();
		void update();
		// void recombine();
		// void catastrophe();

		void crossover(DTYPE *pop1, DTYPE *pop2);
		void select();

		DTYPE LUT(DTYPE alpha, DTYPE beta, DTYPE alphabest, DTYPE betabest);

};

inline double box_muller() {
	double u1 = 1.0 * rand() / RAND_MAX;
	double u2 = 1.0 * rand() / RAND_MAX;
	double result = sqrt(-2.*log(u1)) * cos(2.*M_PI*u2);
	return result;
}

inline double sign(double x) {
	if (x > 0)
		return 1;
	else if (x < 0)
		return -1;
	else
		return 0;
}

#endif
