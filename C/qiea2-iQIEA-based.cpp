
#include "qiea2.h"


void QIEA2::run() {
	t = 0;
	int tmax = 100000 / K;
	initialize();
	observe(); //
	evaluate(); //
	storebest(); //
	while (t < tmax) {
		// printf("Generation %d\n", t);
		// printf("bestval: %f\n", bestval);
		t++;
		observe();
		evaluate();
		storebest();
		update();
	}
}

#define Qij (Q + i * (5 * (chromlen/2)) + (5 * j))
#define Pij (P + i * (chromlen) + (2 * j))

void QIEA2::initialize() {
	bestval = std::numeric_limits<DTYPE>::max();
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen / 2; j++) {
			Qij[0] = (1. * rand() / RAND_MAX) * 200. - 100.;   // X
			Qij[1] = (1. * rand() / RAND_MAX) * 200. - 100.;   // Y
			Qij[2] = M_PI * rand() / RAND_MAX; // orientation
			Qij[3] = 40. * rand() / RAND_MAX; // scale X   XXX 40 param
			Qij[4] = 40. * rand() / RAND_MAX; // scale Y   XXX 40 param
		}
	}
}

void QIEA2::observe() {
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen / 2; j++) {
			double u = Qij[3] * box_muller();
			double v = Qij[4] * box_muller();
			double theta = Qij[2];

			double u2 = u * cos(theta) - v * sin(theta);
			double v2 = u * sin(theta) + v * cos(theta);
			u = u2;
			v = v2;
			u += Qij[0];
			v += Qij[1];
			Pij[0] = u;
			Pij[1] = v;
		}
	}
	// if K > |Q|
	for (int k = popsize; k < K; k++) {
		// choose a random element from Q
		int i = (int)((1. * rand() / RAND_MAX) * popsize);

		// observe it
		for (int j = 0; j < chromlen / 2; j++) {
			double u = Qij[3] * box_muller();
			double v = Qij[4] * box_muller();
			double theta = Qij[2];
			double u2 = u * cos(theta) - v * sin(theta);
			double v2 = u * sin(theta) + v * cos(theta);
			u = u2;
			v = v2;
			u += Qij[0];
			v += Qij[1];

			(P + k * (chromlen) + (2 * j))[0] = u;
			(P + k * (chromlen) + (2 * j))[1] = v;
		}
	}
}

void QIEA2::storebest() {
	DTYPE val = std::numeric_limits<DTYPE>::max();
	int i_best;
	for (int i = 0; i < K; i++) {
		if (fvals[i] < val) { // XXX minmax
			val = fvals[i];
			i_best = i;
		}
	}
	if (val < bestval) { /// XXX minmax
		bestval = val;
		memcpy(best, P + i_best * chromlen, sizeof(DTYPE) * chromlen);
	}
}

void QIEA2::evaluate() {
	for (int i = 0; i < K; i++) {
		fvals[i] = problem->evaluator(P + i * chromlen, chromlen);
	}
}

/*
 * Crossover between pop1 and pop2, and store the result in pop1
 * pop1 and pop2 are size K
 */
void QIEA2::crossover(DTYPE *pop1, DTYPE *pop2) {
	for (int i = 0; i < K; i++) {
		for (int j = 0; j < chromlen; j++) {
			int ind = i * chromlen + j;
			float r = 1.0 * rand() / RAND_MAX;
			float XI = 0.5; // XXX
			if (r < XI) {
				pop1[ind] = pop1[ind];
			}
			else {
				pop1[ind] = pop2[ind];
			}
		}
	}
}

typedef struct {
	int ind;
	QIEA2::DTYPE val;
} elem;

int cmp(const void *p1, const void *p2) {
	elem *elem1 = (elem*)p1;
	elem *elem2 = (elem*)p2;
	if (elem1->val > elem1->val)
		return 1;
	else if (elem1->val < elem1->val)
		return -1;
	return 0;
}

/*
 * Select K best individuals from [C + P] and store it in C
 * fvals needs to store evaluation of P
 * fvalsC needs to store evaluation of C
 */
// select(C,P)
void QIEA2::select() {
	DTYPE result[K];

	// select

	DTYPE both[K * chromlen * 2];
	memcpy(both, C, sizeof(DTYPE) * K * chromlen);
	memcpy(both + K + chromlen, P, sizeof(DTYPE) * K * chromlen);


	elem ranking[2 * K];
	for (int i = 0; i < 2 * K; i++) {
		ranking[i].ind = i;
		ranking[i].val = (i < K) ? fvalsC[i]: fvals[i];
	}
	qsort(ranking, 2*K, sizeof(elem), cmp);

	// copy the result to the destination
	memcpy(C, result, sizeof(DTYPE) * K * chromlen);
}

void QIEA2::update() {
	printf("t: %d\n", this->t);

	if (t < 2) {
		// C <- P
		memcpy(C, P, sizeof(DTYPE) * K * chromlen);
		// XXX sort it?
	}
	else {
		memcpy(fvalsC, fvals, sizeof(DTYPE) * K); // store old fvals
		// P <- cross(P,C)
		crossover(P, C);
		// evaluate current P (store in fvals)
		evaluate();  // -> fvals
		// C <- K best from [C + P]
		select();
	}

	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen / 2; j++) {
			Qij[0] = (C + i * (chromlen) + (2 * j))[2*j];     // XXX zle
			Qij[1] = (C + i * (chromlen) + (2 * j))[2*j + 1]; // XXX zle

			// Qij[2] += .1 // rotating
			Qij[3] *= .9995;
			Qij[4] *= .9995;
		}
	}
}

/*
#include "cec2013.h"
#include <time.h>
int main() {
	//srand(time(0));
	srand(5);//time(0));
	int dim = 10;
	int popsize = 10;
	QIEA2 *qiea2 = new QIEA2(dim, popsize);
	//for (int i = 0; i < dim; i++) {
	//	qiea2->bounds[i][0] = -100;
	//	qiea2->bounds[i][1] = 100;
	//}

	Problem<double,double> *fun = new CEC2013(1);
	// double x[2] = {-39.3, 58.8};
	// double val = fun->evaluator(x, 2);
	// printf("-> %f\n", val);
	// return 0;
	qiea2->problem = fun;
	qiea2->run();
	printf("Final bestval: %f\n", qiea2->bestval);
	printf("Final best: ");
	for (int i = 0; i < dim; i++) {
		printf("%f, ", qiea2->best[i]);
	}
	printf("\n");
}
*/

