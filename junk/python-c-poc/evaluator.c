#include <stdio.h>

int tescik() {
    printf("tescik..\n");
    return 42;
}

double sum(double *x, int dim) {
    int i;
    double result = 0;
    printf("sumowanie\n");
    for (i = 0; i < dim; i++) {
        result += x[i];
    }
    printf("%g\n", result);
    return result;
}

