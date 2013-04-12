
#include "qiea2.h"


void QIEA2::run() {
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

#define Qij (Q + i * (5 * (chromlen/2)) + (5 * j))
#define Pij (P + i * (chromlen) + (2 * j))

void QIEA2::initialize() {
	bestval = std::numeric_limits<DTYPE>::max();
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen / 2; j++) {
			Qij[0] = (1. * rand() / RAND_MAX) * 200. - 100.;   // location X  XXX bounds
			Qij[1] = (1. * rand() / RAND_MAX) * 200. - 100.;   // location Y  XXX bounds
			// Qij[2] = M_PI * rand() / RAND_MAX; // orientation
			Qij[2] = 0; // should be the same as QIEA1
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
}

void QIEA2::storebest() {
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

void QIEA2::evaluate() {
	for (int i = 0; i < popsize; i++) {
		fvals[i] = problem->evaluator(P + i * chromlen, chromlen);
	}
}

void QIEA2::update() {
	/* TUTAJ TRZEBA TO ROZBUDOWAC PRAWIDLOWO */
	for (int i = 0; i < popsize; i++) {
		for (int j = 0; j < chromlen / 2; j++) {
			Qij[0] = best[2*j];
			Qij[1] = best[2*j + 1];
			// Qij[2] += .1 // rotating
			Qij[3] *= .999;
			Qij[4] *= .999;
			continue;





			/* distribution adaptation */

			double anglebest[2] = {
				best[0] * 2. * M_PI / 100, // XXX bounds
				best[1] * 2. * M_PI / 100, // XXX bounds
			};
			Qij[0] += 1. * (2.*M_PI/100) * (anglebest[0] > Qij[0] ? 1 : -1); // XXX bounds
			Qij[1] += 1. * (2.*M_PI/100) * (anglebest[1] > Qij[1] ? 1 : -1); // XXX bounds
			// albo -> do sredniej K najlepszych

			// Qij[2] <- ~ wspolczynnik korelacji??? podejrzec w CMAES???
			// obliczyc to prawidlowo na bazie estymacji
			Qij[2] += fmod(M_PI / 18, (2. * M_PI)); // XXX

			// very simple rule
			Qij[3] *= .98;  // gdy NoFE -> MaxFE, to dx i dy powinny być ~0
			Qij[4] *= .98;
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

