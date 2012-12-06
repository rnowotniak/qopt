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
#include <map>
#include <string>

inline int count(const char *s, int len, char c) {
	int result = 0;
	for (int i = 0; i < len; i++) {
		if (s[i] == c) {
			result++;
		}
	}
	return result;
}

float evaluate_schema(const char *schema, int length, float *mem) {
	const int tries = 100000;
	float fitness = 0.0;
	char buf[length];
	for (int i = 0; i < tries; i++) {
		unsigned x = sample(schema, length);
		fitness += mem[x];
	}
	fitness /= tries;
	//printf("--\n");
	return fitness;
}

/*
 * Szukanie inna metoda -- dla znanego rozwiazania optymalnego x^*
 *
 */
void find_best_schemata(const char *spacefile, const int chromlen) {
	float *mem = new float[int(pow(2,chromlen))];
	FILE *F = fopen(spacefile, "r");
	fread(mem, sizeof(float), int(pow(2, chromlen)), F);
	//printf("-> %f\n", mem[0]);
	//printf("-> %f\n", mem[1]);
	fclose(F);

	long unsigned best = 0;
	float bestval = mem[0];

	for (long unsigned i = 1; i < pow(2,chromlen); i++) {
		if (mem[i] > bestval) {
			best = i;
			bestval = mem[i];
		}
	}

	char beststr[chromlen + 1];
	beststr[chromlen] = '\0';
	dec2bin(beststr, best, chromlen);
	fprintf(stderr, "Best solution possible:\n");
	fprintf(stderr, "%s\n", beststr);
	fprintf(stderr, "-\n");

	char buf[chromlen + 1];
	buf[chromlen] = '\0';
	int maxlen = 5;
	int maxorder = 5;
	int minorder = 3;

	std::map<std::string, float> schemata;

	int masklen = maxlen + 1;
	for (int i = 0; i < chromlen - maxlen; i++) {
		memset(buf, '*', chromlen);
		//memcpy(buf + i, beststr + i, masklen);
		//printf("%s\n", buf);
		//continue;
		for (int j = 0; j < pow(2, masklen); j++) {
			memcpy(buf + i, beststr + i, masklen);
			//printf("%s\n", buf);
			//continue;
			char buf2[masklen + 1];
			buf2[masklen] = '\0';
			dec2bin(buf2, j, masklen);
			int ones = count(buf2, masklen, '1');
			if (ones > maxorder || ones < minorder) {
				continue;
			}
			for (int k = 0; k < masklen; k++) {
				if (buf2[k] == '0') {
					buf[i + k] = '*';
				}
			}
			//printf("%s %d\n", buf2, count(buf2, maxlen, '1'));
			std::string str(buf, chromlen);
			if (schemata.count(str)) {
				continue;
			}
			schemata[str] = evaluate_schema(buf, chromlen, mem);
			//printf("%s\n", str.c_str());
		}
	}

	//printf("-\n");

	for (std::map<std::string, float>::iterator i = schemata.begin();
			i != schemata.end(); i++) {
		printf("%s %f\n", (*i).first.c_str(), (*i).second);
	}

}

int main(int argc, char *argv[]) {

	if (argc != 3) {
		fprintf(stderr, "%s <spacefile> <chromlen>\n", argv[0]);
		return 0;
	}

	srand(time(0));
	find_best_schemata(argv[1], atoi(argv[2]));

	return 0;
}

