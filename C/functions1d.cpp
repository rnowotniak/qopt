#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "interpolation.h"


int bin2dec(char *bin, int len)
{
	int b, k, m, n;
	int sum = 0;
	len -= 1;
	for(k = 0; k <= len; k++)
	{
		n = (bin[k] - '0'); // char to numeric value
		if ((n > 1) || (n < 0))
		{
			puts("\n\n ERROR! BINARY has only 1 and 0!\n");
			return (0);
		}
		for(b = 1, m = len; m > k; m--)
		{
			// 1 2 4 8 16 32 64 ... place-values, reversed here
			b *= 2;
		}
		// sum it up
		sum = sum + n * b;
		//printf("%d*%d + ",n,b); // uncomment to show the way this works
	}
	return(sum);
}

double func1(double x) {
	static int initialized = false;
	static alglib::spline1dinterpolant spline;

	if (initialized == false) {
		alglib::real_1d_array x = "[0, 10, 20, 30, 40, 56, 60, 65, 80, 90, 100, 120, 150, 180, 200]";
		alglib::real_1d_array y = "[0, 20, 40, 10, 25, 33, 80, 45, 60, 20, 0, 20, 40, 20, 0]";
		alglib::spline1dbuildcubic(x, y, spline);
		initialized = true;
	}

	return alglib::spline1dcalc(spline, x);
}

double func2(double x) {
	//     (1.5 * cos(2.*(x-1.5)) + 2.2 * cos(5.*(x-1.5))) * exp(-((x-1.5)/5)**2)
	return (1.5 * cos(2.*(x-1.5)) + 2.2 * cos(5.*(x-1.5))) * exp(-pow((x-1.5)/5., 2));
}

double func3(double x) {
	static int initialized = false;
	static alglib::spline1dinterpolant spline;

	if (initialized == false) {
		alglib::real_1d_array x = "[0, 1, 2, 2.75, 3.4, 4.2, 5, 6, 6.6, 7.2, 8, 9, 9.8, 10.5, 11.2, 13, 14.5, 16.5]";
		alglib::real_1d_array y = "[1.6, 2.3, 2.4, 2.5, -1, 2, 3.3, 3.75, 1.1, 2.2, 4.6, 4.8, 5, .7, 3, 1.5, 4, 3]";
		alglib::spline1dbuildcubic(x, y, spline);
		initialized = true;
	}

	return alglib::spline1dcalc(spline, x);
}

float getx(char *s, int len, float min, float max) {
	float val;
	val = bin2dec(s, len);
	val = min + val * (max - min) / (pow(2, len) - 1);
	return val;
}

float func1_b(char *s,int len) {
	float val = getx(s, len, 0, 200);
	return func1(val);
}
float func2_b(char *s,int len) {
	float val = getx(s, len, -5, 5);
	return func2(val);
}
float func3_b(char *s,int len) {
	float val = getx(s, len, 0, 17);
	return func3(val);
}

