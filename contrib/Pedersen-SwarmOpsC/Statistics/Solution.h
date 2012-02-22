/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Solution
 *
 *	Hold a solution to an optimization problem and its
 *	associated fitness value.
 *
 * ================================================================ */

#ifndef SO_SOLUTION_H
#define SO_SOLUTION_H

#include <SwarmOps/Tools/Types.h>
#include <SwarmOps/Contexts/MethodContext.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	struct SO_Solution
	{
		SO_TElm *x;					/* Position in search-space. */
		SO_TFitness fitness;		/* Associated fitness value. */
		size_t dim;					/* Dimensionality of x. */
	};

	/* Make and return a solution struct holding the best solution and
	 * fitness obtained for the given optimization method context. */
	struct SO_Solution SO_MakeSolution(struct SO_MethodContext *methodContext);

	/* Free the solution. */
	void SO_FreeSolution(struct SO_Solution *s);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_SOLUTION_H */
