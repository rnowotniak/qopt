#ifndef _FUNCTIONS1D_H
#define _FUNCTIONS1D_H 1

#include "framework.h"

extern double func1(double x);
extern double func2(double x);
extern double func3(double x);

extern float func1_b(char *,int);
extern float func2_b(char *,int);
extern float func3_b(char *,int);

extern float getx(char *s, int len, float min, float max);

class Functions1DProblem : public Problem<char,float> {
	public:
	int fnum;
	float (*f)(char *,int);
	Functions1DProblem(int fnum) : fnum(fnum) {
		if (fnum == 1) {
			f = func1_b;
		}
		else if (fnum == 2) {
			f = func2_b;
		}
		else if (fnum == 3) {
			f = func3_b;
		}
	}
	virtual float evaluator (char *x, int length) {
		return f(x, length);
	}
};

#endif
