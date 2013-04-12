/*
 * A very simple algorithm.  My own QIEA1 for Num. opt.
 */

#include "qiea1.h"


void QIEA1::run() {
	t = 0;
	int tmax = 100000 / popsize;
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

#define Qij (Q + i * (2 * chromlen) + (2 * j))
#define Pij (P + i * (chromlen) + (j))

void QIEA1::initialize() {
	bestval = std::numeric_limits<DTYPE>::max();
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen; j++) {
			Qij[0] = (1. * rand() / RAND_MAX) * 200. - 100.;   // mean  XXX bounds
			Qij[1] = (1. * rand() / RAND_MAX) * 40.;   // stddev  XXX bounds
		}
	}
}

void QIEA1::observe() {
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen; j++) {
			double mean = Qij[0];
			double stddev = Qij[1];
			double x = stddev * box_muller() + mean;
			Pij[0] = x;
		}
	}
}

void QIEA1::storebest() {
	DTYPE val = std::numeric_limits<DTYPE>::max();
	int i_best;
	for (int i = 0; i < popsize; i++) {
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

void QIEA1::evaluate() {
	for (int i = 0; i < popsize; i++) {
		fvals[i] = problem->evaluator(P + i * chromlen, chromlen);
	}
}

void QIEA1::update() {
	/* TUTAJ TRZEBA TO ROZBUDOWAC PRAWIDLOWO */
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen; j++) {
			Qij[0] = best[j];
			Qij[1] *= .999;
		}
	}
}

/*
#include "cec2013.h"
#include <time.h>
int main() {
	srand(time(0));
	//srand(5);//time(0));
	int dim = 10;
	int popsize = 10;
	QIEA1 *qiea1 = new QIEA1(dim, popsize);
	//for (int i = 0; i < dim; i++) {
	//	qiea1->bounds[i][0] = -100;
	//	qiea1->bounds[i][1] = 100;
	//}

	Problem<double,double> *fun = new CEC2013(5);
	// double x[2] = {-39.3, 58.8};
	// double val = fun->evaluator(x, 2);
	// printf("-> %f\n", val);
	// return 0;
	qiea1->problem = fun;

	double result = 0.;
	for (int _ = 0; _ < 15; _++) {
		qiea1->run();
		result += qiea1->bestval;
	}
	result /= 15;
	printf("%f\n", result);
	return 0;

	printf("Final bestval: %f\n", qiea1->bestval);
	printf("Final best: ");
	for (int i = 0; i < dim; i++) {
		printf("%f, ", qiea1->best[i]);
	}
	printf("\n");
}
*/

