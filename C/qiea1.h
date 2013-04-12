#ifndef _QIEA1_H
#define _QIEA1_H 1

#include "framework.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>

class QIEA1 {

	public:

		typedef double DTYPE;

		DTYPE *Q; // quantum population
		DTYPE *P; // observed classical population

		int t;

		DTYPE (*bounds)[2];
		int popsize;
		int chromlen;

		DTYPE *fvals;
		DTYPE bestval;
		DTYPE *best;
		// int i_best;
		// DTYPE (*bestq)[2];

		Problem<DTYPE,DTYPE> *problem;

		QIEA1(int chromlen, int popsize) : popsize(popsize), chromlen(chromlen) {
			printf("QIEA1::QIEA1 constructor\n");
			assert(chromlen % 2 == 0);

			problem = NULL;
			bestval = std::numeric_limits<DTYPE>::max();

			Q = new DTYPE[2 * chromlen * popsize];  // (mean,stddev) * chromlen * popsize
			P = new DTYPE[popsize * chromlen];
			fvals = new DTYPE[popsize];
			bounds = new DTYPE[chromlen][2];
			best = new DTYPE[chromlen];
			// bestq = new DTYPE[chromlen][2];
		}

		~QIEA1() {
			delete [] Q;
			delete [] P;
			delete [] bounds;
			delete [] best;
			delete [] fvals;
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
