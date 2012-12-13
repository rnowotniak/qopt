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

// #include "knapsack.h"
#include "qiga.h"


/*
 * The algorithm data
 */

/*
 * Quantum genes initialization stage
 */
void QIGA::initialize() {
	int i, j;
	for (i = 0; i < popsize; i++) {
		for (j = 0; j < chromlen; j++) {
			Qij = M_PI_4; // equal superposition of states
		}
	}
}


/*
 * Observation of classical population stage;
 * Sampling the search space with respect to the quantum population probability distributions
 */
void QIGA::observe() {
	int i, j;
	for (i = 0; i < popsize; i++) {
		for (j = 0; j < chromlen; j++) {
			float alpha = cos(Qij);
			float r = 1.0f * rand() / (RAND_MAX + 1.0);
			P[i][j] = (r < alpha*alpha) ? '0' : '1';
		}
	}
}

void QIGA::repair() {
	for (int i = 0; i < popsize; i++) {
		problem->repairer(P[i], chromlen); // XXX P is modified!
	}
}


/*
 * Individuals evaluation stage
 */
void QIGA::evaluate() {
	int i,j;
	for (i = 0; i < popsize; i++) {
		fvals[i] = problem->evaluator(P[i], chromlen);
	}
}


/*
 * Update stage -- quantum genetic operators; rotations in qubit state spaces
 */
void QIGA::update() {
	int i, j;
	int fxGTfb; // f(x) >= f(b)
	float delta;
	float rot;
	for (i = 0; i < popsize; i++) {
		fxGTfb = fvals[i] >= bestval;
		for (j = 0; j < chromlen; j++) {
			int x = P[i][j];
			int b = best[j];
			delta = lookup_table[x=='1'][b=='1'][fxGTfb];
			float cangle = fmodf(Qij, M_PI); // Qij angle casted into <0,M_PI> interval
			// cf. Table 1 in [1]
			int sindex =
				// if alpha * beta > 0:
				(cangle > EPSILON && cangle < M_PI_2 - EPSILON) ? 0 :
				// if alpha * beta < 0:
				(cangle < M_PI - EPSILON && cangle > M_PI_2 + EPSILON) ? 1 :
				// if alpha == 0:
				(fabsl((fmodf((M_PI_2 + Qij),M_PI) - M_PI_2)) < EPSILON) ? 2 :
				// if beta == 0:
				3;
			float sign = signs_table[x=='1'][b=='1'][fxGTfb][sindex];
			Qij += sign * delta;
		}
	}
}


void QIGA::storebest() {
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

void QIGA::qiga() {
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

