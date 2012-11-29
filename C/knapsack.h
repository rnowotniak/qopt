#ifndef _KNAPSACK_H
#define _KNAPSACK_H 1

#include "framework.h"

// #define items_count 250
// #define CAPACITY 673.8677805
// extern void repairKnapsack(char *x, int length);
// extern float fknapsack(char *k, int length);


class KnapsackProblem : public Problem<char,float> {

    public:

        int items_count;

        float capacity;

        float (*items)[2];

        KnapsackProblem(const char *fname);

        virtual float evaluator(char *x, int length);

        virtual void repairer(char *x, int length);

};

#endif
