/*
 * Quantum-Inspired Genetic Algorithm implementation in C
 * Copyright (C) 2011   Robert Nowotniak <rnowotniak@kis.p.lodz.pl>
 *
 * This implementation replicates Han's results exactly.
 *
 * References:
 *    [1] Han, K.H. and Kim, J.H. Genetic quantum algorithm and its application
 *        to combinatorial optimization problem, 2000
 */

#include "bqigao.h"

#include <cassert>


/*
 * The algorithm data
 */

/*
 * Quantum genes initialization stage
 */
void BQIGAo::initialize() {
	int i, j;
	for (i = 0; i < popsize; i++) {
		for (j = 0; j < chromlen; j++) {
			Qij[0] = M_S2; // equal superposition of states
			Qij[1] = M_S2;
		}
	}
}


/*
 * Observation of classical population stage;
 * Sampling the search space with respect to the quantum population probability distributions
 */
void BQIGAo::observe() {
	int i, j;
	for (i = 0; i < popsize; i++) {
		for (j = 0; j < chromlen; j++) {
			float alpha = Qij[0];
			float r = 1.0f * rand() / (RAND_MAX + 1.0);
			P[i][j] = (r < alpha*alpha) ? '0' : '1';
		}
	}
}

void BQIGAo::repair() {
	for (int i = 0; i < popsize; i++) {
		problem->repairer(P[i], chromlen); // P is modified!
	}
}


/*
 * Individuals evaluation stage
 */
void BQIGAo::evaluate() {
	int i,j;
	for (i = 0; i < popsize; i++) {
		fvals[i] = problem->evaluator(P[i], chromlen);
	}
}


/*
 * Update stage -- quantum genetic operators; rotations in qubit state spaces
 */
void BQIGAo::update() {
	for (int i = 0; i < popsize; i++) {
		int fxGTfb = fvals[i] >= bestval; // f(x) >= f(b)
		for (int j = 0; j < chromlen; j++) {
			int x = P[i][j];
			int b = best[j];
			float delta = lookup_table[x=='1'][b=='1'][fxGTfb];

			// cf. Table 1 in [1]
			int sindex;
			if (Qij[0] * Qij[1] > 0) {
				sindex = 0;
			} else if (Qij[0] * Qij[1] < 0) {
				sindex = 1;
			} else if (Qij[0] == 0) {
				sindex = 2;
			} else if (Qij[1] == 0) {
				sindex = 3;
			}
			else {
				assert(false);
			}

			float sign = signs_table[x=='1'][b=='1'][fxGTfb][sindex];
			float Qprim[2];

			float angle = sign * delta;

			// Q' = U(angle) * Q
			Qprim[0] = Qij[0] * cos(angle) - Qij[1] * sin(angle);
			Qprim[1] = Qij[0] * sin(angle) + Qij[1] * cos(angle);

			// Q <- Q'
			Qij[0] = Qprim[0];
			Qij[1] = Qprim[1];
		}
	}
}


void BQIGAo::storebest() {
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

void BQIGAo::bqigao() {
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
	BQIGAo *b = new BQIGAo(250, 10);
	KnapsackProblem *k = new KnapsackProblem("../problems/knapsack/knapsack-250.txt");
	b->problem = k;
	b->bqigao();
}
*/

