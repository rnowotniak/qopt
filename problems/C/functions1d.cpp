#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "interpolation.h"


alglib::spline1dinterpolant spline;

double func1(double x) {
	static int initialized = false;

	if (initialized == false) {
		alglib::real_1d_array x = "[0, 10, 20, 30, 40, 56, 60, 65, 80, 90, 100, 120, 150, 180, 200]";
		alglib::real_1d_array y = "[0, 20, 40, 10, 25, 33, 80, 45, 60, 20, 0, 20, 40, 20, 0]";
		alglib::spline1dbuildcubic(x, y, spline);
		initialized = true;
	}

	return alglib::spline1dcalc(spline, x);
}

double func2(double x) {
	return 0;
}

double func3(double x) {
	return x + x * x;
}

