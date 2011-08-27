/*
 * Dummy optimization algorithm in C.
 */

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

	int n = 1;
	printf("dummyalgo arguments:\t");
	while (n < argc) {
		printf("%s ", argv[n]);
		n++;
	}
	printf("\n");
	
	int maxgen = 20;

	srand(time(NULL));

	printf("xjzcvkjwekrjkzxcvj ksdjfk awjk ewkrj vzk jkasdfj kwjr ksdj\n");
	printf("vxjkrjwek rjkzxcojvk ewojkrjozcv jokj2o65o zxjco vaosjdf\n");

	int t = 0;
	float tim = 0;
	while (t < maxgen) {
		printf("jfk jzkxcv jaskdf \n");
		printf("vjkasjkej rkaj fkasjf kxzvjk wejkra jsdfk zvkjzsksa djfk\n");
		tim += rand()/(RAND_MAX+1.0);
		float best = 1. + rand()/(RAND_MAX+1.0);
		float worst = rand()/(RAND_MAX+1.0);
		float avg =  worst + (best - worst) * rand()/(RAND_MAX+1.0);
		float sd = 0.2 * rand()/(RAND_MAX+1.0);
		printf("STAT  %d  %d   %g   %g   %g   %g   %g\n",
				t, t * 10, tim, best, avg, worst, sd);
		printf("jfk jzkxcv jaskdf \n");
		printf("vjkasjkej rkaj fkasjf kxzvjk wejkra jsdfk zvkjzsksa djfk\n");
		t++;
	}
	exit(0);

}

