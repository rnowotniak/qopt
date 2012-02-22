/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Meta2Benchmarks.c
 *
 *	Perform meta-meta-optimization experiments using benchmark problems.
 *
 *	Contains the entry-point _tmain(). Should be compilable in
 *	ANSI C, perhaps with minor modifications.
 *
 * ================================================================ */

#include "stdafx.h"

#include <SwarmOps/Meta2OptimizeBenchmarks.h>	/* Meta-meta-optimization using benchmark problems. */
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
	/* Define meta-meta-optimization settings. */

	/* Problem layer having multiple benchmark problems. */
	#define			kNumProblems 5					/* Number of problems. */
	const size_t	kDim = 30;						/* Dimensionality of problems. */
	const int		kDisplaceOptimum = 0;			/* Displace global optimum. */
	const size_t	kProblemIds[kNumProblems] =		/* Array of benchmark problem ID's */
	{
		SO_kBenchmarkSphere,
		SO_kBenchmarkGriewank,
		SO_kBenchmarkRastrigin,
		SO_kBenchmarkAckley,
		SO_kBenchmarkRosenbrock
	};

	/* Optimization layer having multiple optimization methods. */
	#define			kNumMethods 2					/* Number of optimization methods. */
	const size_t	kMethodId[kNumMethods] =		/* Array of method ID's */
	{
		SO_kMethodPSO,
		SO_kMethodDE
	};
	const size_t	kNumRuns = 50;						/* Number of optimization runs per method. */
	const size_t	kDimFactor = 200;					/* Dimensionality factor. */
	const size_t	kNumIterations = kDimFactor*kDim;	/* Number of iterations per run. */

	/* Meta-optimization layer. */
	const size_t kMetaMethodId = SO_kMethodLUS;			/* Meta-optimizer. */
	const size_t kMetaNumRuns = 10;						/* Number of meta-optimization runs. */
	const size_t kMetaDimFactor = 20;					/* Dimensionality factor. */

	/* Meta-meta-optimization layer. */
	const size_t kMeta2MethodId = SO_kMethodLUS;						/* Meta-meta-optimizer. */
	const size_t kMeta2NumRuns = 1;										/* Number of meta-meta-opt. runs. */
	const size_t kMeta2DimFactor = 25;									/* Dimensionality factor. */
	const size_t kMeta2Dim = SO_kMethodNumParameters[kMetaMethodId];	/* Dimensionality (ie. number of parameters to meta-meta-optimize. */
	const size_t kMeta2NumIterations = kMeta2DimFactor * kMeta2Dim;		/* Number of iterations. */

	/* Solution-struct used for holding results of meta-optimization
	 * until they can be printed. */
	struct SO_Solution s;

	/* Timing variables. */
	clock_t t1, t2;

	/* Iteration variable. */
	size_t i;

	/* Array holding number of iterations to perform for each optimization method. */
	size_t metaNumIterations[kNumMethods];

	/* Initialize array. */
	for (i=0; i<kNumMethods; i++)
	{
		/* Get the number of parameters for the i'th method, this is the
		 * dimensionality of the meta-optimization problem. */
		size_t metaDim = SO_kMethodNumParameters[kMethodId[i]];

		/* The number of iterations per meta-optimization run equals
		 * the appropriate dim-factor times the number of parameters
		 * for the given optimization method. */
		metaNumIterations[i] = kMetaDimFactor * metaDim;
	}

	/* Display meta-meta-optimization settings. */
	printf("Meta-Meta-Optimization of benchmark problems.\n");
	printf("\n");
	printf("Meta-meta-method: %s\n", SO_kMethodName[kMeta2MethodId]);
	printf("Using following parameters:\n");
	SO_PrintParameters(kMeta2MethodId, SO_kMethodDefaultParameters[kMeta2MethodId]);
	printf("Number of meta-meta-runs: %i\n", kMeta2NumRuns);
	printf("Number of meta-meta-iterations: %i\n", kMeta2NumIterations);
	printf("\n");
	printf("Meta-method: %s\n", SO_kMethodName[kMetaMethodId]);
	printf("Number of meta-runs: %i\n", kMetaNumRuns);
	printf("\n");

	for (i=0; i<kNumMethods; i++)
	{
		printf("Method no. %i to be meta-optimized: %s\n", i+1, SO_kMethodName[kMethodId[i]]);
		printf("Number of meta-iterations: %i\n", metaNumIterations[i]);
		printf("\n");
	}

	printf("Number of benchmark problems: %i\n", kNumProblems);
	printf("Dimensionality for each benchmark problem: %i\n", kDim);
	printf("Number of runs per benchmark problem: %i\n", kNumRuns);
	printf("Number of iterations per run: %i\n", kNumIterations);
	printf("Displace global optimum: %s\n", (kDisplaceOptimum) ? ("Yes") : ("No"));
	printf("Init. in full search-space: No");
	printf("\n");

	printf("*** Indicates a meta-meta-fitness evaluation is an improvement.\n\n");

	/* Seed the pseudo-random number generator. */
	RO_RandSeedClock(89691245);

	/* Timing start. */
	t1 = clock();

	/* Perform meta-meta-optimization. */
	s = SO_Meta2OptimizeBenchmarks(
		kMeta2MethodId, kMeta2NumRuns, kMeta2NumIterations, 0,
		kMetaMethodId, kMetaNumRuns, metaNumIterations, 0,
		kNumMethods, kMethodId, kNumRuns, kNumIterations, 0,
		kProblemIds, kNumProblems, kDim, kDisplaceOptimum);

	/* Timing end. */
	t2 = clock();

	/* Display results of meta-meta-optimization along with time-usage. */
	printf("\nTime usage: %g seconds\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
	printf("Best parameters for %s used as meta-optimizer:\n", SO_kMethodName[kMetaMethodId]);
	SO_PrintParameters(kMetaMethodId, s.x);
	printf("Parameters written in vector notation:\n");
	SO_PrintVector(s.x, s.dim);
	printf("\n");
	printf("With meta-meta-fitness: %g\n", s.fitness);

	/* Free the contents of the solution-struct. */
	SO_FreeSolution(&s);

	return 0;
}

/* ---------------------------------------------------------------- */
