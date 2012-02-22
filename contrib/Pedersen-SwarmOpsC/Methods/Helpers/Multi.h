/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Multi
 *
 *	Allows for one optimization method to be used on
 *	multiple problems. The results are summed to form a
 *	total fitness value. It requires one method-context
 *	per problem.
 *
 *	Note that the multiple problems are sorted so as to
 *	make the one that is expected to yield worst fitness
 *	be computed first. This improves Pre-Emptive Fitness
 *	Evaluation further still.
 *
 * ================================================================ */

#ifndef SO_MULTI_H
#define SO_MULTI_H

#include <SwarmOps/Contexts/MethodContext.h>
#include <SwarmOps/Tools/Types.h>
#include <stddef.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Internal struct used in the sorting of the problems. */
	struct SO_MultiIndex;

	/* Struct holding data on the optimization method and the multiple
	 * problems, as well as evaluation order. */
	struct SO_MultiContext
	{
		SO_FMethod method;					/* Optimization method. */
		void **contexts;					/* Method-contexts, one for each problem. */
		size_t numProblems;					/* Number of problems. */
		struct SO_MultiIndex* ordering;		/* Evaluation ordering. */
	};

	/* Create and return a MultiContext. Arguments are as described for the struct above. */
	struct SO_MultiContext SO_MakeMultiContext(SO_FMethod method, void **contexts, size_t numProblems);

	/* Free memory allocated by SO_MakeMultiContext(), but not memory
	 * allocated elsewhere, such as the array of method-contexts. */
	void SO_FreeMultiContext(struct SO_MultiContext *c);

	/* ---------------------------------------------------------------- */

	/* Function computing and returning the total fitness of using the designated
	 * method to optimize the designated problems. */
	SO_TFitness SO_Multi(const SO_TElm *param, void *context, const SO_TFitness fitnessLimit);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_MULTI_H */
