/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	BenchmarkContext
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Contexts/BenchmarkContext.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

struct SO_BenchmarkContext
	SO_MakeBenchmarkContext
	(SO_TDim n, int displaceOptimum)
{
	struct SO_BenchmarkContext c;

	c.n = n;
	c.displaceOptimum = displaceOptimum;

	return c;
}

/* ---------------------------------------------------------------- */
