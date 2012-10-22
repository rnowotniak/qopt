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

#include "knapsack.h"
#include "qiga.h"


/*
 * The algorithm data
 */
// Lookup table: rotation angles in Qubit state spaces



/*
 * Individuals evaluation stage
 */
void QIGA::evaluate() {
	int i,j;
	for (i = 0; i < popsize; i++) {
		fvals[i] = ((evaluator_t)evaluator)(P[i]);
	}
}

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

void QIGA::repair() {
	int i;
	for (i = 0; i < popsize; i++) {
		repairKnapsack(P[i]);
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
	while (t < maxgen) {
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


int main() {
	printf("QiGA on CPU\n");

	struct timeval start_tm;
	gettimeofday(&start_tm, 0);
	srand(time(0)+start_tm.tv_usec);
	//srand(1298823875);

	QIGA *qiga = new QIGA();

	int rep;
	for (rep = 0; rep < REPEAT; rep++) {
		fprintf(stderr, ".");
		qiga->qiga();
	}
	fprintf(stderr, "\n");

	struct timeval stop_tm;
	gettimeofday(&stop_tm, 0);
	printf("%g seconds\n", (1e6 * (stop_tm.tv_sec - start_tm.tv_sec) + (stop_tm.tv_usec - start_tm.tv_usec))/1e6);

	return 0;
}

