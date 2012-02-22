/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MetaOptimizeBenchmarks
 *
 *	Function for performing meta-optimization with regard
 *	to multiple benchmark problems. This is a basic
 *	wrapper-function for MetaOptimizeMulti.
 *
 * ================================================================ */

#ifndef SO_METAOPTIMIZEBENCHMARKS_H
#define SO_METAOPTIMIZEBENCHMARKS_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Statistics/Solution.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Meta-optimize the behavioural parameters of a given method with
	 * respect to multiple benchmark problems, using a meta-optimizer. */
	struct SO_Solution SO_MetaOptimizeBenchmarks

		/* Meta-optimization layer. */
		(size_t metaMethodId,			/* Meta-optimization method (see Methods.h) */
		size_t metaNumRuns,				/* Number of meta-optimization runs. */
		size_t metaNumIterations,		/* Number of iterations per meta-optimization run. */
		void *metaSettings,				/* Additional meta-optimization settings. */

		/* Optimization layer. */
		size_t methodId,				/* Optimization method whose parameters are to be tuned. */
		size_t numRuns,					/* Number of runs of the optimization method. */
		size_t numIterations,			/* Number of iterations per optimization run (same for all problems). */
		void *settings,					/* Additional optimization settings (same for all problems). */

		/* Multiple-problems layer. */
		const size_t *problemIds,		/* Benchmark problems to be optimized (see Benchmarks.h) */
		size_t numProblems,				/* Number of problems to be optimized. */
		SO_TDim n,						/* Dimensionality of the problems to be optimized (same for all). */
		int displaceOptimum,			/* Displace global optimum for all problems. */
		int initFullRange,				/* Use full search-space for initialization range. */

		/* Other arguments. */
		const char *traceFilename);		/* Meta-fitness-trace filename (null-pointer for no trace). */

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_METAOPTIMIZEBENCHMARKS_H */
