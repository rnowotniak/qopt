/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Benchmarks2.c
 *
 *	Perform optimization experiments on benchmark problems.
 *
 *	Contains the entry-point _tmain(). Should be compilable in
 *	ANSI C, perhaps with minor modifications.
 *
 * ================================================================ */

#include "stdafx.h"

#include <SwarmOps/OptimizeBenchmark.h>			/* Convenient function for optimizing benchmark problems. */
#include <RandomOps/Random.h>					/* Pseudo-random number generator, for seeding. */

#include <SwarmOps/Methods/Methods.h>			/* Optimization method ID-handles. */
#include <SwarmOps/Problems/Benchmarks.h>		/* Benchmark problem ID-handles. */

#include <stdlib.h>
#include <time.h>

/* ---------------------------------------------------------------- */

const size_t	kMethodId = SO_kMethodDE;		/* Optimization method. */
const size_t	kNumRuns = 50;					/* Number of optimization runs. */
const size_t	kDim = 30;						/* Problem dimensionality. */
const size_t	kDimFactor = 200;				/* Iterations per run = dim * kDimFactor */
const int		kDisplaceOptimum = 0;			/* Displace global optimum. */
const int		kInitFullRange = 0;				/* Initialize in full search-space (easier to optimize). */

/* ---------------------------------------------------------------- */

/* Accumulation of optimization results for the benchmark problems. */
SO_TFitness		gFitnessMinTotal = 0;
SO_TFitness		gFitnessMaxTotal = 0;
SO_TFitness		gFitnessAvgTotal = 0;
SO_TFitness		gFitnessStdDevTotal = 0;

/* ---------------------------------------------------------------- */

/* Helper function for doing optimization on a problem.
 * Prints the results to std.out. */
void Benchmark(const size_t kMethodId, const size_t kProblemId)
{
	const size_t kNumIterations = kDimFactor*kDim;

	const char* fitnessTraceName = 0; /* "FitnessTrace-DErand1binJitter-Meta3Bnch-Sphere.txt"; */

	struct SO_Statistics stat;

	printf("%s & ", SO_kBenchmarkName[kProblemId]);

	stat = SO_OptimizeBenchmark(kMethodId, kNumRuns, kNumIterations, 0, kProblemId, kDim, kDisplaceOptimum, kInitFullRange, fitnessTraceName);

	printf("%g & %g & %g & %g \\\\\n", stat.fitnessMin, stat.fitnessMax, stat.fitnessAvg, stat.fitnessStdDev);

	gFitnessMinTotal += stat.fitnessMin;
	gFitnessMaxTotal += stat.fitnessMax;
	gFitnessAvgTotal += stat.fitnessAvg;
	gFitnessStdDevTotal += stat.fitnessStdDev;
}

/* ---------------------------------------------------------------- */

/* Main entry-point for the console application.
 * You may need to rename this to compile with other compilers
 * than MS Visual C++, for instance:
 * int main(int argc, char* argv[])
 */

int _tmain(int argc, _TCHAR* argv[])
{
	/* Timing variables. */
	clock_t t1, t2;

	/* Display optimization settings to std.out. */
	printf("Benchmark-tests.\n");
	printf("Method: %s\n", SO_kMethodName[kMethodId]);
	printf("Using following parameters:\n");
	SO_PrintParameters(kMethodId, SO_kMethodDefaultParameters[kMethodId]);
	printf("Number of runs per problem: %i\n", kNumRuns);
	printf("Dimensionality: %i\n", kDim);
	printf("Dim-factor: %i\n", kDimFactor);
	printf("Displace global optimum: %s\n", (kDisplaceOptimum) ? ("Yes") : ("No"));
	printf("Init. in full search-space: %s\n", (kInitFullRange) ? ("Yes") : ("No"));
	printf("\n");
	printf("Problem & Best & Worst & Average & Std.Dev. \\\\\n");
	printf("\\hline\n");

	/* Seed the pseudo-random number generator. */
	RO_RandSeedClock(43455783);

	/* Timer start. */
	t1 = clock();

	/* Perform optimizations on different benchmark problems.
	 * Display the results during optimization. */
	Benchmark(kMethodId, SO_kBenchmarkSphere);
	Benchmark(kMethodId, SO_kBenchmarkSchwefel222);
	Benchmark(kMethodId, SO_kBenchmarkSchwefel12);
	Benchmark(kMethodId, SO_kBenchmarkSchwefel221);
	Benchmark(kMethodId, SO_kBenchmarkRosenbrock);
	Benchmark(kMethodId, SO_kBenchmarkStep);
	Benchmark(kMethodId, SO_kBenchmarkQuarticNoise);
	Benchmark(kMethodId, SO_kBenchmarkRastrigin);
	Benchmark(kMethodId, SO_kBenchmarkAckley);
	Benchmark(kMethodId, SO_kBenchmarkGriewank);
	Benchmark(kMethodId, SO_kBenchmarkPenalized1);
	Benchmark(kMethodId, SO_kBenchmarkPenalized2);

	/* Timing end. */
	t2 = clock();

	printf("\\hline\n");
	printf("Sum & %g & %g & %g & %g \\\\\n", gFitnessMinTotal, gFitnessMaxTotal, gFitnessAvgTotal, gFitnessStdDevTotal);

	/* Display time-usage to std.out. */
	printf("\nTime usage: %g seconds\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

	return 0;
}

/* ---------------------------------------------------------------- */
