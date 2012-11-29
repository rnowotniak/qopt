#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <strings.h>
#include <time.h>

#include "framework.h"
#include "functions1d.h"
#include "sat.h"
#include "knapsack.h"


#include <vector>

void find_best_schemata() {
	const int chromlen = 25;
	//const char *schemata_filename = "../../data/schem15-uniq";
	//std::vector<char[chromlen+1]> schem;

	int number_of_schemata = 8683;
	int number_of_chromosomes = 100;

	char (*schemata)[chromlen + 1];
	char (*chromosomes)[chromlen + 1];
	float *schemata_fitness;
	float *chromosomes_fitness;

	// allocate memory for the above data structures
	schemata = new char[number_of_schemata][chromlen + 1];
	chromosomes = new char[number_of_chromosomes][chromlen + 1];
	schemata_fitness = new float[number_of_schemata];
	chromosomes_fitness = new float[int(pow(2,chromlen))];

	// load data
	{
		char line[256];
		FILE *F;
		F = fopen("../../data/schem25-uniq", "r");
		int i = 0;
		while (true) {
			if (fgets(line, sizeof(line), F) == NULL) {
				break;
			}
			line[chromlen] = '\0';
			memcpy(schemata[i++], line, chromlen + 1);
		}
		fclose(F);

		F = fopen("../../data/func1d-25-best", "r");
		i = 0;
		while (i < number_of_chromosomes) {
			if (fgets(line, sizeof(line), F) == NULL) {
				break;
			}
			unsigned numb = atoi(line);
			dec2bin(line, numb, chromlen);
			line[chromlen] = '\0';
			memcpy(chromosomes[i++], line, chromlen + 1);
		}
		fclose(F);

		F = fopen("../../data/func1d-25-space", "r");
		fread((void*)chromosomes_fitness, sizeof(float), pow(2, chromlen), F);
		fclose(F);
	}

	// evaluate schemata
	for (int i = 0; i < number_of_schemata; i++) {
		int number_of_matching_chromosomes = 0;
		schemata_fitness[i] = 0.0;
		for (int j = 0; j < number_of_chromosomes; j++) {
			if (matches(chromosomes[j], schemata[i], chromlen)) {
				unsigned idx = strtoul(chromosomes[j], NULL, 2);
				schemata_fitness[i] += chromosomes_fitness[idx];
				number_of_matching_chromosomes++;
			}
		}
		if (number_of_matching_chromosomes == 0) {
			continue;
		}
		while (number_of_matching_chromosomes < 1000) {
			unsigned idx = sample(schemata[i], chromlen);
			schemata_fitness[i] += chromosomes_fitness[idx];
			number_of_matching_chromosomes++;
		}
		schemata_fitness[i] /= 1000.;
	}

	// print schemata and their fitness
	for (int i = 0; i < number_of_schemata; i++) {
		printf("%s %f\n", schemata[i], schemata_fitness[i]);
	}

	delete [] schemata;
	delete [] chromosomes;
	delete [] schemata_fitness;
	delete [] chromosomes_fitness;
}

int main() {

	srand(time(0));

	find_best_schemata();

}

