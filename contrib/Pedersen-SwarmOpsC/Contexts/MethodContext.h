/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	MethodContext
 *
 *	The context-struct used for optimization methods. The
 *	struct holds data such as the number of iterations,
 *	the problem to be optimized, the problem's context,
 *	boundaries, etc.
 *
 *	Examples of optimization methods using this: DE and PSO.
 *
 * ================================================================ */

#ifndef SO_METHODCONTEXT_H
#define SO_METHODCONTEXT_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Statistics/FitnessTrace.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* The context-struct for optimization methods. */
	struct SO_MethodContext
	{
		SO_FProblem f;					/* The fitness function to be optimized. */
		SO_FGradient fGradient;			/* Gradient for fitness function. Can be zero. */
		void *fContext;					/* The problem's context. */
		SO_TDim fDim;					/* Dimensionality of problem. */
		SO_TElm const* lowerInit;		/* Lower initialization boundary. */
		SO_TElm const* upperInit;		/* Upper initialization boundary. */
		SO_TElm const* lowerBound;		/* Lower search-space boundary. */
		SO_TElm const* upperBound;		/* Upper search-space boundary. */
		size_t numIterations;			/* Number of fitness-evaluations. */

		SO_TElm* g;						/* Best position for current run. */
		SO_TFitness gFitness;			/* Fitness for best position in current run. */

		SO_TElm* bestPosition;			/* Alltime best found solution to problem. */
		SO_TFitness bestFitness;		/* Fitness for best-found position. */

		void *settings;					/* Extra settings your method may require. */

		struct SO_FitnessTrace *fitnessTrace;		/* Fitness trace. */
	};

	/* ---------------------------------------------------------------- */

	/* Make and return a context-struct.
	 * Parameters correspond to the struct description above. */
	struct SO_MethodContext SO_MakeMethodContext
		(SO_FProblem f,
		SO_FGradient fGradient,
		void const* fContext,
		SO_TDim fDim,
		SO_TElm const* lowerInit,
		SO_TElm const* upperInit,
		SO_TElm const* lowerBound,
		SO_TElm const* upperBound,
		size_t numIterations,
		void *settings,
		struct SO_FitnessTrace *fitnessTrace);

	/* Free memory allocated by SO_MakeMethodContext(),
	 * but not memory for the struct itself, the boundaries, etc. */
	void SO_FreeMethodContext(struct SO_MethodContext *c);

	/*----------------------------------------------------------------*/

	/* Update the best-position and -fitness stored in context
	 * if the new fitness is an improvement (i.e. fitness is less). */
	void SO_MethodUpdateBest
		(struct SO_MethodContext *c,
		const SO_TElm *newPosition,
		SO_TFitness newFitness);

	/*----------------------------------------------------------------*/

	/* Set the final result of the current optimization run. */
	void SO_MethodSetResult
		(struct SO_MethodContext *c,
		const SO_TElm *position,
		SO_TFitness fitness);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_METHODCONTEXT_H */

/* ================================================================ */
