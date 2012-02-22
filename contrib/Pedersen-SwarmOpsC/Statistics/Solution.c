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
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Statistics/Solution.h>
#include <SwarmOps/Tools/Vector.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_Solution SO_MakeSolution(struct SO_MethodContext *methodContext)
{
	struct SO_Solution s;

	s.x = SO_NewVector(methodContext->fDim);
	SO_CopyVector(s.x, methodContext->bestPosition, methodContext->fDim);
	s.fitness = methodContext->bestFitness;
	s.dim = methodContext->fDim;

	return s;
}

/* ---------------------------------------------------------------- */

void SO_FreeSolution(struct SO_Solution *s)
{
	SO_FreeVector(s->x);
	s->x = 0;
}

/* ---------------------------------------------------------------- */
