#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define BLOCK 40
#define REPEAT 2500
#define DIM 50

double rosenbrock (double *x, int dim)
{
    int i;
    double res;
    res = 0.0;
    for (i=0; i<dim-1; i++)
    {
        res += 100.0*pow((x[i]*x[i]-x[i+1]),2.0) + 1.0*pow((x[i]-1.0),2.0);
    }
    return (res);
}

double cec2005complex(double *in, int dim) {
    int i;
    float x;
    float y;
    for (i = 0; i < 1e6; i++) {
        x = 5.55;
        x = x + x;
        x = x / 2;
        x = x * x;
        x = sqrt(x);
        x = log(x);
        x = exp(x);
        y = x/x;
    }
    return y;
}

int main() {
    struct timeval tv1, tv2;
    struct timezone tz1, tz2;
    float tdiff;
    int i,j;
    double DATA[BLOCK][DIM];

    srand(2011);

    for (i = 0; i < BLOCK; i++) {
        for (j = 0; j < DIM; j++) {
            DATA[i][j] = (1.0 * rand() / RAND_MAX) * 60 - 30;
        }
    }
    // data.txt
//    for (i = 0; i < BLOCK; i++) {
//        for (j = 0; j < DIM; j++) {
//            printf("%lf ", DATA[i][j]);
//        }
//        printf("\n");
//    }
//    // values.txt
//    for (i = 0; i < BLOCK; i++) {
//        printf("%lf\n", cec2005complex(((double*)DATA) + i * DIM, DIM));
//    }
//    return 0;

    double values[BLOCK];

    int evaluations = 0;

    gettimeofday(&tv1, &tz1);
    for (j = 0; j < REPEAT; j++) {
        for (i = 0; i < BLOCK; i++) {
            values[i] = rosenbrock(((double*)DATA) + i * DIM, DIM);
	    evaluations++;
        }
    }
    gettimeofday(&tv2, &tz2);

    tdiff = (tv2.tv_sec - tv1.tv_sec) * 1e6 + 1.0f * (tv2.tv_usec - tv1.tv_usec);
    printf("evals: %d\n", evaluations);
    printf("%f microseconds\n", tdiff);
    printf("performance: %f (evals / microsecond)\n", 1.0 * BLOCK * REPEAT / tdiff);
}

