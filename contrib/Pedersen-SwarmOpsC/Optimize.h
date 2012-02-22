/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Optimize
 *
 *	Wrapper-functions for optimizing a given problem.
 *
 * ================================================================ */

#ifndef SO_OPTIMIZE_H
#define SO_OPTIMIZE_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Statistics/Results.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Optimize the given problem. */
	struct SO_Results SO_Optimize(
		size_t methodId,				/* Optimization method. */
		size_t numRuns,					/* Number of optimization runs. */
		size_t numIterations,			/* Number of iterations per run. */
		void *settings,					/* Additional optimization settings. */
		SO_FProblem f,					/* Optimization problem (aka. fitness function). */
		SO_FGradient fGradient,			/* Gradient for optimization problem. */
		void *fContext,					/* Context for optimization problem. */
		SO_TDim n,						/* Dimensionality for optimization problem. */
		SO_TElm const* lowerInit,		/* Lower initialization boundary. */
		SO_TElm const* upperInit,		/* Upper initialization bounder. */
		SO_TElm const* lowerBound,		/* Lower search-space boundary. */
		SO_TElm const* upperBound,		/* Upper search-space boundary. */
		const char *traceFilename);		/* Fitness trace filename (null-pointer for no trace). */

	/* Same as SO_Optimize() but using custom parameters for optimization method. */
	struct SO_Results SO_OptimizePar(
		SO_TElm const* par,				/* Behavioural parameters for optimization method. */
		size_t methodId,
		size_t numRuns,
		size_t numIterations,
		void *settings,
		SO_FProblem f,
		SO_FGradient fGradient,
		void *fContext,
		SO_TDim n,
		SO_TElm const* lowerInit,
		SO_TElm const* upperInit,
		SO_TElm const* lowerBound,
		SO_TElm const* upperBound,
		const char *traceFilename);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_OPTIMIZE_H */
