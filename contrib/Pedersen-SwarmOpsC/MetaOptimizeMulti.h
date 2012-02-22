/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MetaOptimizeMulti
 *
 *	Function for performing meta-optimization with regard
 *	to multiple optimization problems. That is, to find
 *	the behavioural parameters of an optimization method which
 *	makes it perform its best on the given optimization problems.
 *	To do this, another overlaid optimization method is used,
 *	which is referred to as the Meta-Optimizer.
 *
 *	This function is similar to MetaOptimize(), except that
 *	this allows for the tuning of behavioural parameters with
 *	regard to multiple optimization problems.
 *
 * ================================================================ */

#ifndef SO_METAOPTIMIZEMULTI_H
#define SO_METAOPTIMIZEMULTI_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Statistics/Solution.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Meta-optimize the behavioural parameters of a given method with
	 * respect to multiple problems, using the designated meta-optimizer. */
	struct SO_Solution SO_MetaOptimizeMulti

		/* Meta-optimization layer. */
		(size_t metaMethodId,			/* Meta-optimization method (see Methods.h) */
		size_t metaNumRuns,				/* Number of meta-optimization runs. */
		size_t metaNumIterations,		/* Number of iterations per meta-optimization run. */
		void *metaSettings,				/* Additional meta-optimization settings. */

		/* Optimization layer. */
		size_t methodId,				/* Optimization method whose parameters are to be tuned. */
		size_t numRuns,					/* Number of runs of the optimization method. */
		size_t *numIterations,			/* Number of iterations per optimization run (one for each problem). */
		void *settings,					/* Additional optimization settings (same for all problems). */

		/* Multiple-problems layer. */
		size_t numProblems,				/* Number of problems to be optimized. */
		SO_FProblem *f,					/* Problems to be optimized. */
		SO_FGradient *fGradient,		/* Gradients of the problems to be optimized. */
		void **contexts,				/* Contexts of the problems to be optimized. */
		SO_TDim *dim,					/* Dimensionalities of the problems to be optimized. */
		SO_TElm **lowerInit,			/* Lower initialization boundaries. */
		SO_TElm **upperInit,			/* Upper initialization boundaries. */
		SO_TElm **lowerBound,			/* Lower search-space boundaries. */
		SO_TElm **upperBound,			/* Upper search-space boundaries. */

		/* Other arguments. */
		const char *traceFilename);		/* Meta-fitness-trace filename (null-pointer for no trace). */

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_METAOPTIMIZEMULTI_H */
