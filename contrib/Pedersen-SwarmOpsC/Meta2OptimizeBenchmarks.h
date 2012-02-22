/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Meta2OptimizeBenchmarks
 *
 *	Function for performing meta-meta-optimization with
 *	regard to multiple optimizatino methods and multiple
 *	benchmark problems. This is a basic wrapper-function
 *	for Meta2OptimizeMulti.
 *
 * ================================================================ */

#ifndef SO_META2OPTIMIZEBENCHMARKS_H
#define SO_META2OPTIMIZEBENCHMARKS_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Statistics/Solution.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Meta-meta-optimize the behavioural parameters of a given meta-optimizer
	 * with respect to multiple methods and benchmark problems. */
	struct SO_Solution SO_Meta2OptimizeBenchmarks

		/* Meta-meta-optimization layer. */
		(size_t meta2MethodId,				/* Meta-meta-optimization method. */
		size_t meta2NumRuns,				/* Number of meta-meta-optimization runs. */
		size_t meta2NumIterations,			/* Number of iterations per meta-meta-optimization run. */
		void *meta2Settings,				/* Additional meta-meta-optimization settings. */

		/* Meta-optimization layer. */
		size_t metaMethodId,				/* Meta-optimization method (see Methods.h) */
		size_t metaNumRuns,					/* Number of meta-optimization runs. */
		const size_t *metaNumIterations,	/* Number of iterations per meta-optimization run. */
		void *metaSettings,					/* Additional meta-optimization settings. */

		/* Optimization layer (multiple methods supported). */
		size_t numMethods,					/* Number of optimization methods. */
		const size_t *methodId,				/* Optimization methods. */
		size_t numRuns,						/* Number of optimization runs per method. */
		size_t numIterations,				/* Number of iterations per run (same for all problems). */
		void **settings,					/* Additional optimization settings (one for each method). */

		/* Problem layer (multiple benchmark problems supported). */
		const size_t *problemIds,			/* Benchmark problems to be optimized (see Benchmarks.h) */
		size_t numProblems,					/* Number of problems to be optimized. */
		SO_TDim n,							/* Dimensionality of the problems to be optimized (same for all). */
		int displaceOptimum);				/* Displace global optimum for all problems. */

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_META2OPTIMIZEBENCHMARKS_H */
