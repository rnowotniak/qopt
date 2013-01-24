#ifndef _QIGA_H
#define _QIGA_H 1

/*
 * This implementation is based on trigonometric (angular) representation
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <sys/time.h>
#include "framework.h"

#define EPSILON 10e-9f
#undef M_PI
#undef M_PI_2
#undef M_PI_4
#define M_PI	3.14159265358979323846f	/* pi float */
#define M_PI_2	1.57079632679489661923f	/* pi/2 float */
#define M_PI_4	0.78539816339744830962f	/* pi/4 float */


class QIGA {

    public:

	int tmax;

	const int popsize;
	const int chromlen;

	float *Q;
	char **P;
	float *fvals;
	char *best;

	float bestval;

	// Lookup table: rotation angles in Qubit state spaces
	float lookup_table[2][2][2]; // [x][b][f(x)>=f(b)]

	// Rotation directions
	float signs_table[2][2][2][4]; // [x][b][f(x)>=f(b)][s(alpha*beta)]

	Problem<char,float> *problem;

	QIGA(int chromlen, int popsize) : popsize(popsize), chromlen(chromlen) {
		printf("QIGA::QIGA constructor\n");

		tmax = 500;

		problem = NULL;

		Q = new float[popsize * chromlen];
		fvals = new float[popsize];
		best = new char[chromlen];

		P = new char*[popsize];
		for (int i = 0; i < popsize; i++) {
			P[i] = new char[chromlen + 1];
			P[i][chromlen] = '\0';
		}

		float lt[2][2][2] = 
		{
			0, 0, 0,
			0.05 * M_PI, 0.01 * M_PI, 0.025 * M_PI, 0.005 * M_PI, 0.025 * M_PI
		};
		float st[2][2][2][4] =
		{
			0,  0,  0,  0,
			0,  0,  0,  0,
			0,  0,  0,  0,
			-1, +1, +1,  0,
			-1, +1, +1,  0,
			+1, -1,  0, +1,
			+1, -1,  0, +1,
			+1, -1,  0, +1,
		};
		memcpy(this->lookup_table, lt, sizeof(lt));
		memcpy(this->signs_table, st, sizeof(st));
	}

	~QIGA() {
		for (int i = 0; i < popsize; i++) {
			delete [] P[i];
		}
		delete P;
		delete Q;
		delete fvals;
		delete best;
	}

	// complete algorithm in C++
	void qiga();

	// elementary operations
	void evaluate();
	void initialize();
	void observe();
	void storebest();
	void update();
	void repair();

};

#endif
