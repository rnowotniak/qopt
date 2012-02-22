/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Bound
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Tools/Types.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

SO_TElm SO_Bound(const SO_TElm x, const SO_TElm lower, const SO_TElm upper)
{
	SO_TElm y = x;

	assert(upper >= lower);

	if (y < lower)
	{
		y = lower;
	}
	else if (y > upper)
	{
		y = upper;
	}

	return y;
}

/* ---------------------------------------------------------------- */

void SO_BoundOne(SO_TElm *x, SO_TDim i, const SO_TElm *lower, const SO_TElm *upper)
{
	assert(upper[i] >= lower[i]);

	if (x[i] < lower[i])
	{
		x[i] = lower[i];
	}
	else if (x[i] > upper[i])
	{
		x[i] = upper[i];
	}
}

/* ---------------------------------------------------------------- */

void SO_BoundAll(SO_TElm *x, SO_TDim n, const SO_TElm *lower, const SO_TElm *upper)
{
	SO_TDim i;

	for (i=0; i<n; i++)
	{
		SO_BoundOne(x, i, lower, upper);
	}
}

/* ---------------------------------------------------------------- */
