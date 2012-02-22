/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MetaOptimize
 *
 *	Function for performing meta-optimization. That is, to find
 *	the behavioural parameters of an optimization method which
 *	makes it perform its best on the given optimization problem.
 *	To do this, another overlaid optimization method is used,
 *	which is referred to as the Meta-Optimizer.
 *
 * ================================================================ */

#ifndef SO_METAOPTIMIZE_H
#define SO_METAOPTIMIZE_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Statistics/MetaSolution.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Meta-optimize the behavioural parameters of a given method
	 * with respect to one problem, using the designated meta-optimizer. */
	struct SO_MetaSolution SO_MetaOptimize

		/* Meta-optimization layer. */
		(size_t metaMethodId,			/* Meta-optimization method (see Methods.h) */
		size_t metaNumRuns,				/* Number of meta-optimization runs. */
		size_t metaNumIterations,		/* Number of iterations per meta-optimization run. */
		void *metaSettings,				/* Additional meta-optimization settings. */

		/* Optimization layer. */
		size_t methodId,				/* Optimization method whose parameters are to be tuned. */
		size_t numRuns,					/* Number of runs of the optimization method. */
		size_t numIterations,			/* Number of iterations per optimization run. */
		void *settings,					/* Additional optimization settings. */

		/* Problem layer. */
		SO_FProblem f,					/* Problem to be optimized. */
		SO_FGradient fGradient,			/* Gradient of the problem to be optimized. */
		void const* fContext,			/* Context of the problem to be optimized. */
		SO_TDim n,						/* Dimensionality of the problem to be optimized. */
		SO_TElm const* lowerInit,		/* Lower initialization boundary. */
		SO_TElm const* upperInit,		/* Upper initialization boundary. */
		SO_TElm const* lowerBound,		/* Lower search-space boundary. */
		SO_TElm const* upperBound);		/* Upper search-space boundary. */

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_METAOPTIMIZE_H */
