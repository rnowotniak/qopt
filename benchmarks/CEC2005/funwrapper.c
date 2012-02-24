#include <stdio.h>

#include "global.h"
#include "rand.h"

long double evaluate(long double *x, int n) {
    static int initialized_dimension = -1;

    if (initialized_dimension != n) {
        initialized_dimension = n;
        printf("Initializing for dimension %d\n", n);

        nreal = n;

#if f15 || f16 || f17 || f18 || f19 || f20 || f21 || f22 || f23 || f24 || f25
        nfunc = 10;
#else
        nfunc = 2;
#endif

        randomize();
        initrandomnormaldeviate();
        allocate_memory();
        initialize();
#if f15 || f16 || f17 || f18 || f19 || f20 || f21 || f22 || f23 || f24 || f25
        calc_benchmark_norm();
#endif
    }

    return calc_benchmark_func(x);
}

