#ifndef _QIGA_H
#define _QIGA_H 1

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cstring>
#include <sys/time.h>

#define EPSILON 10e-9f
#undef M_PI
#undef M_PI_2
#undef M_PI_4
#define M_PI	3.14159265358979323846f	/* pi float */
#define M_PI_2	1.57079632679489661923f	/* pi/2 float */
#define M_PI_4	0.78539816339744830962f	/* pi/4 float */

//#define chromlen 250
//#define popsize 10

#define Qij (Q[i][j])

#ifndef REPEAT
#define REPEAT 100
#endif

/*
void qiga(void);
extern float bestval;
void initialize(void);
void observe(void);
void repair(void);
void evaluate(void);
void update(void);
void storebest(void);
*/


class QIGA {

    public:

	int popsize;
	int chromlen;
	int maxgen;

	/*
	float Q[popsize][chromlen];
	char P[popsize][chromlen];
	float fvals[popsize];
	char best[chromlen];
	*/

	float **Q;
	char **P;
	float *fvals;
	char *best;

	float bestval;

	float lookup_table[2][2][2]; // [x][b][f(x)>=f(b)]

	// Rotation directions
	float signs_table[2][2][2][4]; // [x][b][f(x)>=f(b)][s(alpha*beta)]

	QIGA() {
		maxgen = 500;
		popsize = 10;
		chromlen = 250;

		Q = new float*[popsize];
		P = new char*[popsize];
		for (int i = 0; i < popsize; i++) {
			Q[i] = new float[chromlen];
			P[i] = new char[chromlen];
		}

		fvals = new float[popsize];
		best = new char[chromlen];

		float lt[2][2][2] = 
		{
			0,
			0,
			0,
			0.05 * M_PI,
			0.01 * M_PI,
			0.025 * M_PI,
			0.005 * M_PI,
			0.025 * M_PI
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
			delete [] Q[i];
			delete [] P[i];
		}
		delete Q;
		delete P;
		delete fvals;
		delete best;
	}

	void evaluate();
	void initialize();
	void observe();
	void storebest();
	void update();
	void repair();
	void qiga();

};

#endif
