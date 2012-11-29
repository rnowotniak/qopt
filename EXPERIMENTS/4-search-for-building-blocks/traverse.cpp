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

inline void dec2bin(char *buf, long int dec, int length) {
	memset(buf, '0', length);
	int i = length - 1;
	while (dec > 0) {
		buf[i] = '0' + dec % 2;
		dec = dec >> 1;
		i--;
	}
}

inline void dec2ternary(char *buf, long int dec, int length) {
	memset(buf, '0', length);
	int i = length - 1;
	while (dec > 0) {
		buf[i] = '0' + dec % 3;
		if (buf[i] == '2') {
			buf[i] = '*';
		}
		dec = dec / 3;
		i--;
	}
}

inline int order(char *s, int len) {
	int result = 0;
	for (int i = 0; i < len; i++) {
		if (s[i] != '*') {
			result++;
		}
	}
	return result;
}

inline int deflenth(char *s, int len) {
	int start = -1, stop;
	for (int i = 0; i < len; i++) {
		if (s[i] != '*') {
			start = i;
			break;
		}
	}
	if (start == -1) {
		return -1;
	}
	for (int i = len - 1; i >= 0; i--) {
		if (s[i] != '*') {
			stop = i;
			break;
		}
	}
	return stop - start;
}

/*
inline bool matches(const char *chromo, const char *schema, int len) {
	for (int i = 0; i < len; i++) {
		if (schema[i] != '*' && schema[i] != chromo[i]) {
			return false;
		}
	}
	return true;
}
*/

#include <vector>

inline unsigned sample(const char *schema, int length) {
	char s[length];
	for (int i = 0; i < length; i++) {
		if (schema[i] != '*') {
			s[i] = schema[i];
			continue;
		}
		s[i] = ((1.f * rand() / RAND_MAX) > .5) ? '1' : '0';
	}
	return strtoul(s, NULL, 2);
}

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

		F = fopen("../../data/k25-best", "r");
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

		F = fopen("../../data/k25-space", "r");
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

