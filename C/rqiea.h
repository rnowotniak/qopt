#ifndef _RQIEA_H
#define _RQIEA_H 1

#include "framework.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

class RQIEA {

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
		DTYPE (*bestq)[2];

		Problem<DTYPE,DTYPE> *problem;

		RQIEA(int chromlen, int popsize) : popsize(popsize), chromlen(chromlen) {
			printf("RQIEA::RQIEA constructor\n");

			problem = NULL;

			Q = new DTYPE[popsize * chromlen * 2];
			P = new DTYPE[popsize * chromlen];
			fvals = new DTYPE[popsize];
			bounds = new DTYPE[chromlen][2];
			best = new DTYPE[chromlen];
			bestq = new DTYPE[chromlen][2];
		}

		~RQIEA() {
			delete [] Q;
			delete [] P;
			delete [] bounds;
			delete [] best;
			delete [] fvals;
			delete [] bestq;
		}

		// the whole algorithm in C++
		void run();

		// elementary operations
		void initialize();
		void observe();
		void storebest();
		void evaluate();
		void update();
		void recombine();
		void catastrophe();

		DTYPE LUT(DTYPE alpha, DTYPE beta, DTYPE alphabest, DTYPE betabest);

};

inline double sign(double x) {
	if (x > 0)
		return 1;
	else if (x < 0)
		return -1;
	else
		return 0;
}

#endif
