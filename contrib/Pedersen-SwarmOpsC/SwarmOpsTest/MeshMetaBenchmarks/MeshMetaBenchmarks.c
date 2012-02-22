/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MeshMetaBenchmarks.c
 *
 *	Compute a mesh of performance values using benchmark problems,
 *	where the performance measure is the same as used in meta-optimization.
 *
 *	Contains the entry-point _tmain(). Should be compilable in
 *	ANSI C, perhaps with minor modifications.
 *
 * ================================================================ */

#include "stdafx.h"

#include <SwarmOps/MetaOptimizeBenchmarks.h>	/* Meta-optimization using benchmark problems. */
#include <RandomOps/Random.h>					/* Pseudo-random number generator, for seeding. */

#include <SwarmOps/Methods/Methods.h>			/* Optimization method ID-handles. */
#include <SwarmOps/Problems/Benchmarks.h>		/* Benchmark problem ID-handles. */
#include <SwarmOps/Tools/Vector.h>				/* Vector operations, such as print. */

#include <stdlib.h>
#include <time.h>

/* ---------------------------------------------------------------- */

/* Main entry-point for the console application.
 * You may need to rename this to compile with other compilers
 * than MS Visual C++, for instance:
 * int main(int argc, char* argv[])
 */

int _tmain(int argc, _TCHAR* argv[])
{
	/* Define meta-optimization settings. */

	/* Problem layer having multiple benchmark problems. */
	const size_t	kDim = 20;						/* Dimensionality of problems. */
	const int		kDisplaceOptimum = 0;			/* Displace global optimum (bool). */
	const int		kInitFullRange = 0;				/* Use full search-space for init. (bool). */
	#define			kNumProblems 12					/* Number of problems. */
	const size_t	kProblemIds[kNumProblems] =		/* Array of benchmark problem ID's */
	{
		SO_kBenchmarkSphere,
		SO_kBenchmarkSchwefel222,
		SO_kBenchmarkSchwefel12,
		SO_kBenchmarkSchwefel221,
		SO_kBenchmarkGriewank,
		SO_kBenchmarkPenalized1,
		SO_kBenchmarkPenalized2,
		SO_kBenchmarkRastrigin,
		SO_kBenchmarkAckley,
		SO_kBenchmarkRosenbrock,
		SO_kBenchmarkStep,
		SO_kBenchmarkQuarticNoise
	};

	/* Optimization layer. */
	const size_t	kMethodId = SO_kMethodDETP;			/* Optimization method to be meta-optimized. */
	const size_t	kNumRuns = 50;						/* Number of optimization runs. */
	const size_t	kDimFactor = 200;					/* Dimensionality factor. */
	const size_t	kNumIterations = kDimFactor*kDim;	/* Number of iterations per run. */

	/* Mesh details. */
	const size_t kMeshNumIterations = 20000;			/* Total number of mesh points. */

	/* Solution-struct used for holding results of meta-optimization
	 * until they can be printed. */
	struct SO_Solution s;

	/* Timing variables. */
	clock_t t1, t2;

	/* Display meta-optimization settings. */
	printf("Mesh of performance values for Meta-Optimization of benchmark problems.\n");
	printf("\n");
	printf("Number of points in mesh (approx.): %i\n", kMeshNumIterations);
	printf("\n");
	printf("Method to be meta-optimized: %s\n", SO_kMethodName[kMethodId]);
	printf("Number of benchmark problems: %i\n", kNumProblems);
	printf("Dimensionality for each benchmark problem: %i\n", kDim);
	printf("Number of runs per benchmark problem: %i\n", kNumRuns);
	printf("Number of iterations per run: %i\n", kNumIterations);
	printf("Displace global optimum: %s\n", (kDisplaceOptimum) ? ("Yes") : ("No"));
	printf("Init. in full search-space: %s\n", (kInitFullRange) ? ("Yes") : ("No"));
	printf("\n");

	/* Seed the pseudo-random number generator. */
	RO_RandSeedClock(73543564);

	/* Timing start. */
	t1 = clock();

	/* Perform meta-optimization. */
	s = SO_MetaOptimizeBenchmarks(
		SO_kMethodMESH, 1, kMeshNumIterations, 0,
		kMethodId, kNumRuns, kNumIterations, 0,
		kProblemIds, kNumProblems, kDim, kDisplaceOptimum, kInitFullRange,
		0);

	/* Timing end. */
	t2 = clock();

	/* Display results of meta-optimization along with time-usage. */
	printf("\nTime usage: %g seconds\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
	printf("Best found parameters for %s optimizer:\n", SO_kMethodName[kMethodId]);
	SO_PrintParameters(kMethodId, s.x);
	printf("Parameters written in vector notation:\n");
	SO_PrintVector(s.x, s.dim);
	printf("\n");
	printf("With fitness: %g\n", s.fitness);

	/* Free the contents of the solution-struct. */
	SO_FreeSolution(&s);

	return 0;
}

/* ---------------------------------------------------------------- */
