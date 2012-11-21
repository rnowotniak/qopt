#ifndef _KNAPSACK_H
#define _KNAPSACK_H 1

#include "framework.h"

#define items_count 250
#define CAPACITY 673.8677805

extern float items[items_count][2];

extern void repairKnapsack(char *x, int length);
extern float fknapsack(char *k, int length);

class KnapsackProblem : public Problem<char,float> {
	virtual float evaluator (char *x, int length) {
		return fknapsack(x, length);
	}
	virtual void repairer (char *x, int length) {
		repairKnapsack(x, length);
	}
//	virtual long double r_evaluator(long double *x, int length) {
//		return -1;
//	}
};

#endif
