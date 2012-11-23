#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <strings.h>

#include "../../framework.h"
#include "../../problems/C/functions1d.h"
#include "../../problems/C/sat.h"

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

int main() {
	{
		long int i, bound;
		int len = 15;
		char buf[len + 1];
		memset(buf, '\0', len + 1);
		bound = pow(3, len);
		for (i = 0; i < bound; i++) {
			dec2ternary(buf, i, len);
			int o = order(buf, len);
			if (o > 3 || o < 2) {
				continue;
			}
			int dl = deflenth(buf, len);
			if (dl > 2) {
				continue;
			}
			printf("%5d %s %2d %2d\n", i, buf, o, dl);
		}
	}
	return 0;

	long int i;
	int chromlen = 25;
	char buf[chromlen + 1];
	char best[chromlen + 1];
	memset(buf, '\0', chromlen + 1);
	memset(best, '\0', chromlen + 1);
	//SAT *s3 = new SAT("../../contrib/SPY-1.2/formula.tmp.cnf");
	SAT *s3 = new SAT("../../problems/sat/random-20.cnf");
	int max = -1;
	printf("%d\n", s3->numclause);
	float *mem = new float[int(pow(2,chromlen))];
	long int topbound = pow(2,chromlen);
	for (i = 0; i < topbound; i++) {
		dec2bin(buf, i, chromlen);
		int val = s3->evaluator(buf, chromlen);
		mem[i] = val;
		if (val > max) {
			max = val;
			memcpy(best, buf, chromlen);
			// break;
		}
		//printf("%5d %s   ", i, buf);
		//printf("%d\n", val);
	}
	int occurences = 0;
	for (i = 0; i < topbound; i++) {
		if (mem[i] == max) {
			occurences++;
		}
	}
	fprintf(stderr, "max: %d\n", max);
	fprintf(stderr, "occurences: %d\n", occurences);
	fprintf(stderr, "%s   \n", best);
	//printf("aa\n");
}

