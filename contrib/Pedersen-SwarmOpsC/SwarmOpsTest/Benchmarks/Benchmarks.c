/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Benchmarks.c
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
const size_t	kDimFactor = 200;				/* Iterations per run = dim * kDimFactor */
const int		kDisplaceOptimum = 0;			/* Displace global optimum. */
const int		kInitFullRange = 0;				/* Initialize in full search-space (easier to optimize). */

/* ---------------------------------------------------------------- */

/* Helper function for actually doing optimization on a problem.
 * Prints the results to std.out. */
void DoBenchmark(const size_t kMethodId, const size_t kProblemId, SO_TDim kDim)
{
	const size_t kNumIterations = kDimFactor*kDim;

	const char* fitnessTraceName = 0; /* "FitnessTrace-Sphere100-LUS.txt"; */

	struct SO_Statistics stat = SO_OptimizeBenchmark(kMethodId, kNumRuns, kNumIterations, 0, kProblemId, kDim, kDisplaceOptimum, kInitFullRange, fitnessTraceName);

	printf("%g (%g)", stat.fitnessAvg, stat.fitnessStdDev);
}

/* ---------------------------------------------------------------- */

/* Helper function for doing optimization on a problem with different
 * dimensionalities. */
void Benchmark(const size_t kMethodId, const size_t kProblemId)
{
	printf("%s & ", SO_kBenchmarkName[kProblemId]);

	DoBenchmark(kMethodId, kProblemId, 20);

	printf(" & ");

	DoBenchmark(kMethodId, kProblemId, 50);

	printf(" & ");

	DoBenchmark(kMethodId, kProblemId, 100);

	printf(" \\\\\n");
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
	printf("Benchmark-tests in various dimensions.\n");
	printf("Method: %s\n", SO_kMethodName[kMethodId]);
	printf("Using following parameters:\n");
	SO_PrintParameters(kMethodId, SO_kMethodDefaultParameters[kMethodId]);
	printf("Number of runs per problem: %i\n", kNumRuns);
	printf("Dim-factor: %i\n", kDimFactor);
	printf("Displace global optimum: %s\n", (kDisplaceOptimum) ? ("Yes") : ("No"));
	printf("Init. in full search-space: %s\n", (kInitFullRange) ? ("Yes") : ("No"));
	printf("\n");
	printf("Problem & 20 dim. & 50 dim. & 100 dim. \\\\\n");

	/* Seed the pseudo-random number generator. */
	RO_RandSeedClock(9385839);

	/* Timer start. */
	t1 = clock();

	/* Perform optimizations on different benchmark problems.
	 * Display the results during optimization. */
	Benchmark(kMethodId, SO_kBenchmarkSphere);
	Benchmark(kMethodId, SO_kBenchmarkGriewank);
	Benchmark(kMethodId, SO_kBenchmarkRastrigin);
	Benchmark(kMethodId, SO_kBenchmarkAckley);
	Benchmark(kMethodId, SO_kBenchmarkRosenbrock);

	/* Timing end. */
	t2 = clock();

	/* Display time-usage to std.out. */
	printf("\nTime usage: %g seconds\n", (double)(t2 - t1) / CLOCKS_PER_SEC);

	return 0;
}

/* ---------------------------------------------------------------- */
