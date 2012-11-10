#include "stdafx.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "interpolation.h"

using namespace alglib;


int main(int argc, char **argv)
{
    //
    // We use cubic spline to interpolate f(x)=x^2 sampled 
    // at 5 equidistant nodes on [-1,+1].
    //
    // First, we use default boundary conditions ("parabolically terminated
    // spline") because cubic spline built with such boundary conditions 
    // will exactly reproduce any quadratic f(x).
    //
    // Then we try to use natural boundary conditions
    //     d2S(-1)/dx^2 = 0.0
    //     d2S(+1)/dx^2 = 0.0
    // and see that such spline interpolated f(x) with small error.
    //
    real_1d_array x = "[-1.0,-0.5,0.0,+0.5,+1.0]";
    real_1d_array y = "[+1.0,0.25,0.0,0.25,+1.0]";
    x = "[0, 10, 20, 30, 40, 56, 60, 65, 80, 90, 100, 120, 150, 180, 200]";
    y = "[0, 20, 40, 10, 25, 33, 80, 45, 60, 20, 0, 20, 40, 20, 0]";
    double t = 0.25;
    double v;
    spline1dinterpolant s;
    ae_int_t natural_bound_type = 2;
    //
    // Test exact boundary conditions: build S(x), calculare S(0.25)
    // (almost same as original function)
    //
    spline1dbuildcubic(x, y, s);
    //v = spline1dcalc(s, t);
    //printf("%.4f\n", double(v)); // EXPECTED: 0.0625

    for (float X = 0; X <= 200; X += .01) {
	    v = spline1dcalc(s, X);
	    printf("%g %.4f\n", X, double(v)); // EXPECTED: 0.0625
    }

    //
    // Test natural boundary conditions: build S(x), calculare S(0.25)
    // (small interpolation error)
    //
//     spline1dbuildcubic(x, y, 5, natural_bound_type, 0.0, natural_bound_type, 0.0, s);
//     v = spline1dcalc(s, t);
//     printf("%.3f\n", double(v)); // EXPECTED: 0.0580

    return 0;
}

