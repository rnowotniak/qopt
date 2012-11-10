#ifndef _KNAPSACK_H
#define _KNAPSACK_H 1

#define items_count 250
#define CAPACITY 673.8677805

extern float items[items_count][2];

extern void repairKnapsack(char *x, int length);
extern float fknapsack(char *k, int length);

#endif
