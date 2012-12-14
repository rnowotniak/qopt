/*
 * Quantum-Inspired Genetic Algorithm type II implementation in C
 * Copyright (C) 2012   Robert Nowotniak <rnowotniak@kis.p.lodz.pl>
 *
 */

#include "bqigao2.h"

#define Qij (Q + i * (4 * (chromlen/2)) + (4 * j))

/*
 * The algorithm data
 */

/*
 * Quantum genes initialization stage
 */
void BQIGAo2::initialize() {
	int i, j;
	for (i = 0; i < popsize; i++) {
		for (j = 0; j < chromlen/2; j++) { // XXX ok???
			Qij[0] = .5f; // equal superposition of states
			Qij[1] = .5f;
			Qij[2] = .5f;
			Qij[3] = .5f;
		}
	}
}


/*
 * Observation of classical population stage;
 * Sampling the search space with respect to the quantum population probability distributions
 */
void BQIGAo2::observe() {
	int i, j;
	for (i = 0; i < popsize; i++) {
		for (j = 0; j < chromlen/2; j++) { // XXX ???
			float r = 1.0f * rand() / (RAND_MAX + 1.0);
			if (r < Qij[0]*Qij[0]) {
				memcpy(&(P[i][2*j]), "00", 2);
			}
			else if (r < Qij[0]*Qij[0] + Qij[1]*Qij[1]) {
				memcpy(&(P[i][2*j]), "01", 2);
			}
			else if (r < Qij[0]*Qij[0] + Qij[1]*Qij[1] + Qij[2]*Qij[2]) {
				memcpy(&(P[i][2*j]), "10", 2);
			}
			else { // r < P(00) + P(01) + P(10) + P(11) = 1
				memcpy(&(P[i][2*j]), "11", 2);
			}
		}
	}
}

void BQIGAo2::repair() {
	for (int i = 0; i < popsize; i++) {
		problem->repairer(P[i], chromlen); // P is modified!
	}
}


/*
 * Individuals evaluation stage
 */
void BQIGAo2::evaluate() {
	int i,j;
	for (i = 0; i < popsize; i++) {
		fvals[i] = problem->evaluator(P[i], chromlen);
	}
}


/*
void kronecker(float result[4][4], float m1[2][2], float m2[2][2]) {
	
}
*/

/*
 * Update stage -- quantum genetic operators; rotations in qubit state spaces
 */
void BQIGAo2::update() {
	for (int i = 0; i < popsize; i++) {
		int fxGTfb = fvals[i] >= bestval; // f(x) >= f(b)
		for (int j = 0; j < chromlen / 2; j++) { // XXX ???
			// int x = P[i][j]; // wrong j
			// int b = best[j]; // wrong j

			// Q'
			float Qprim[4];

			// update Q
			char buf[3] = { '\0' };
			memcpy(buf, best + (j * 2), 2);
			int bestamp = strtol(buf, 0, 2);
			float sum = 0.f;
			for (int amp = 0; amp < 4; amp++) {
				if (amp != bestamp) {
					Qprim[amp] = this->mi * Qij[amp];
					sum += Qprim[amp] * Qprim[amp];
				}
			}
			Qprim[bestamp] = sqrtf(1.f - sum);

			// Q <- Q'
			Qij[0] = Qprim[0];
			Qij[1] = Qprim[1];
			Qij[2] = Qprim[2];
			Qij[3] = Qprim[3];
		}
	}
}


void BQIGAo2::storebest() {
	int i;
	float val = -1;
	char *b;
	for (i = 0; i < popsize; i++) {
		if (fvals[i] > val) {
			val = fvals[i];
			b = P[i];
		}
	}
	if (val > bestval) {
		bestval = val;
		memcpy(best, b, chromlen);
	}
}



/*
   void show() {
   int i,j;
   for (i = 0; i < popsize; i++) {
   for (j = 0; j < chromlen; j++) {
   printf("%c", P[i][j]);
   }
   printf("   %f\n", fvals[i]);
   }
   }
   */

void BQIGAo2::bqigao2() {
	int t = 0;
	bestval = -1;
	initialize();
	observe();
	repair();
	evaluate();
	storebest();
	while (t < tmax) {
		//printf("generation %d\n", t);
		observe();
		repair();
		evaluate();
		update();
		storebest();
		//show();
		t++;
	}

	printf("best solution: ");
	fwrite(best, 1, chromlen, stdout);
	printf("\nfitness: %f\n", bestval);
	fflush(stdout);
}

/*
#include "knapsack.h"

int main() {
	srand(time(0));
	BQIGAo2 *b = new BQIGAo2(250, 10);
	KnapsackProblem *k = new KnapsackProblem("../problems/knapsack/knapsack-250.txt");
	b->problem = k;
	b->bqigao2();
}
*/

