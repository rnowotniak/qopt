/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Init
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Tools/Init.h>
#include <SwarmOps/Tools/Random.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

void SO_InitUniform(SO_TElm *x, SO_TDim n, const SO_TElm *lower, const SO_TElm *upper)
{
	SO_TDim i;

	for (i=0; i<n; i++)
	{
		SO_TElm l = lower[i];
		SO_TElm u = upper[i];

		assert(u >= l);

		x[i] = SO_RandBetween(l, u);
	}
}

/* ---------------------------------------------------------------- */

void SO_InitRange(SO_TElm *d, SO_TDim n, const SO_TElm *lower, const SO_TElm *upper)
{
	SO_TDim i;

	for (i=0; i<n; i++)
	{
		d[i] = upper[i]-lower[i];

		assert(d[i] >= 0);
	}
}

/* ---------------------------------------------------------------- */
