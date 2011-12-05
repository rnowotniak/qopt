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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define EPSILON 10e-9f
#undef M_PI
#undef M_PI_2
#undef M_PI_4
#define M_PI	3.14159265358979323846f	/* pi float */
#define M_PI_2	1.57079632679489661923f	/* pi/2 float */
#define M_PI_4	0.78539816339744830962f	/* pi/4 float */

#define chromlen 250
#define popsize 10
const int maxgen = 500;
float Q[popsize][chromlen];
char P[popsize][chromlen];
float fvals[popsize];
char best[chromlen];
float bestval;

#define Qij (Q[i][j])

float lookup_table[2][2][2] = { // [x][b][f(x)>=f(b)]
	0, 0, 0, // constant, not tuned

	//0, 0.038, 0.349, 0.334, 0.349
	0.0, 0.013563103127078785, 0.16582640019020597, 0.008273234080406186, 0.19293470412099775 // mf2 = 3281 (very good) (level = 1450)

	//0.098016, 0.088810, 0.224968, 0.230300, 0.083285, 0.104184, 0.195625, 0.055723
	//0.075663, 0.333653, 0.272377, 0.063604, 0.038765, 0.239345, 0.326676, 0.320071   // (tuned)
	//0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
	//0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
	//0, 0, 0, 0.05 * M_PI, 0.01 * M_PI, 0.025 * M_PI, 0.005 * M_PI, 0.025 * M_PI
};
float signs_table[2][2][2][4] = {// [x][b][f(x)>=f(b)][s(alpha*beta)]
	 0,  0,  0,  0,
	 0,  0,  0,  0,
	 0,  0,  0,  0,
	-1, +1, +1,  0,
	-1, +1, +1,  0,
	+1, -1,  0, +1,
	+1, -1,  0, +1,
	+1, -1,  0, +1,
};

#include "items.h"

// repair procedure exactly from Han's paper
void repairKnapsack(char *x) {
	float weight;
	int overfilled;
	int i,j;
	weight = 0;
	for (i = 0; i < items_count; i++) {
		weight += items[i][0] * (x[i] == '1' ? 1 : 0);
	}
	overfilled = weight > CAPACITY;
	for (i = 0; i < items_count && overfilled; i++) {
		weight -= x[i] == '1' ? items[i][0] : 0;
		x[i] = '0';
		overfilled = weight > CAPACITY;
	}
	for (j = 0; j < items_count && overfilled == 0; j++) {
		weight += x[j] == '0' ? items[j][0] : 0;
		x[j] = '1';
		overfilled = weight > CAPACITY;
	}
	x[j-1] = '0';
}
float fknapsack(char *k) {
	int i;
	float price = 0; // total price of k knapsack
	for (i = 0; i < items_count; i++) {
		price += items[i][1] * (k[i] == '1' ? 1 : 0);
	}
	return price;
}
/* }}} KNAPSACK */


void evaluate() {
	int i,j;
	for (i = 0; i < popsize; i++) {
		fvals[i] = fknapsack(P[i]);
	}
}

void initialize() {
	int i, j;
	for (i = 0; i < popsize; i++) {
		for (j = 0; j < chromlen; j++) {
			Qij = M_PI_4;
		}
	}
}

void observe() {
	int i, j;
	for (i = 0; i < popsize; i++) {
		for (j = 0; j < chromlen; j++) {
			float alpha = cos(Qij);
			float r = 1.0f * rand() / (RAND_MAX + 1.0);
			P[i][j] = (r < alpha*alpha) ? '0' : '1';
		}
	}
}

void storebest() {
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

void update() {
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

void repair() {
	int i;
	for (i = 0; i < popsize; i++) {
		repairKnapsack(P[i]);
	}
}

void show() {
	int i,j;
	for (i = 0; i < popsize; i++) {
		for (j = 0; j < chromlen; j++) {
			printf("%c", P[i][j]);
		}
		printf("   %f\n", fvals[i]);
	}
}

void qiga() {
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
        printf("(%d) %f\n", t, bestval);
		storebest();
		//show();
		t++;
	}

	printf("best solution: ");
	fwrite(best, 1, chromlen, stdout);
	printf("fitness: %f\n", bestval);
	fflush(stdout);
}

#ifndef REPEAT
#define REPEAT 50
#endif

int main() {
	printf("QiGA on CPU\n");

	struct timeval start_tm;
	gettimeofday(&start_tm, 0);
	srand(time(0)+start_tm.tv_usec);
	//srand(1298823875);

	int rep;
	for (rep = 0; rep < REPEAT; rep++) {
		qiga();
	}

	struct timeval stop_tm;
	gettimeofday(&stop_tm, 0);
	fprintf(stderr,"%g seconds\n", (1e6 * (stop_tm.tv_sec - start_tm.tv_sec) + (stop_tm.tv_usec - start_tm.tv_usec))/1e6);

	return 0;
}

