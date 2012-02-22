/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	OptimizeBenchmark
 *
 *	Functions for optimizing benchmark problems.
 *	This is implemented as a simple wrapper for
 *	the Optimize() function.
 *
 * ================================================================ */

#ifndef SO_OPTIMIZEBENCHMARK_H
#define SO_OPTIMIZEBENCHMARK_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Statistics/Statistics.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Optimize the given benchmark problem using the given method. */
	struct SO_Statistics SO_OptimizeBenchmark	(
		size_t methodId,				/* Optimization method. */
		size_t numRuns,					/* Number of optimization runs. */
		size_t numIterations,			/* Number of iterations per run. */
		void *settings,					/* Additional optimization settings. */
		size_t problemId,				/* Benchmark problem id (see Benchmarks.h) */
		SO_TDim n,						/* Dimensionality of problem. */
		int displaceOptimum,			/* Displace global optimum of problem. */
		int initFullRange,				/* Initialize agents in full search-space. */
		const char *traceFilename);		/* Fitness trace filename (null pointer for none). */

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_OPTIMIZEBENCHMARK_H */
