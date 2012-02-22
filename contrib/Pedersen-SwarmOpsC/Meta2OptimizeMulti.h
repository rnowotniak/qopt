/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Meta2OptimizeMulti
 *
 *	Function for doing meta-meta-optimization, that is,
 *	to find the best-performing behavioural parameters
 *	of a meta-optimizer.
 *
 *	There are several conceptual layers involved:
 *	-	The Meta-Meta-Optimizer which finds parameters
 *		of the Meta-Optimizer.
 *	-	The Meta-Optimizer which finds parameters of
 *		the Optimizer.
 *	-	The Optimizer which finds solutions to an
 *		actual optimization problem.
 *
 *	Furthermore, multiple base-layer problems are allowed
 *	to cause good generalization of the parameters discovered
 *	by the Meta-Optimizer. And multiple Optimizers are
 *	also allowed to cause good parameters discovered by the
 *	Meta-Meta-Optimizer.
 *
 * ================================================================ */

#ifndef SO_META2OPTIMIZEMULTI_H
#define SO_META2OPTIMIZEMULTI_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Statistics/Solution.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Meta-meta-optimize the behavioural parameters of a given meta-optimizer
	 * with respect to multiple methods and problems. */
	struct SO_Solution SO_Meta2OptimizeMulti

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
		const size_t *numIterations,		/* Number of iterations per run (one for each problem). */
		void **settings,					/* Additional optimization settings (one for each method). */

		/* Problem layer (multiple problems supported). */
		size_t numProblems,				/* Number of problems to be optimized. */
		SO_FProblem *f,					/* Problems to be optimized. */
		SO_FGradient *fGradient,		/* Gradients of the problems to be optimized. */
		void **contexts,				/* Contexts of the problems to be optimized. */
		SO_TDim *dim,					/* Dimensionalities of the problems to be optimized. */
		SO_TElm **lowerInit,			/* Lower initialization boundaries. */
		SO_TElm **upperInit,			/* Upper initialization boundaries. */
		SO_TElm **lowerBound,			/* Lower search-space boundaries. */
		SO_TElm **upperBound);			/* Upper search-space boundaries. */

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_META2OPTIMIZEMULTI_H */
