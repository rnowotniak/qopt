/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Denormal
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Tools/Types.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

/* If fabs(x) is above a limit return x and hence not denormalized,
 * otherwise return zero. */

SO_TElm SO_DenormalTrunc(SO_TElm x)
{
	/* The limit below which the number x is considered denormalized. */
	const double kDenormalLimit =  1e-30;

	return (fabs(x) > kDenormalLimit) ? (x) : (0);
}

/* ---------------------------------------------------------------- */

void SO_DenormalFixOne(SO_TElm *x, SO_TDim i)
{
	x[i] = SO_DenormalTrunc(x[i]);
}

/* ---------------------------------------------------------------- */

void SO_DenormalFixAll(SO_TElm *x, SO_TDim n)
{
	size_t i;

	for (i=0; i<n; i++)
	{
		SO_DenormalFixOne(x, i);
	}
}

/* ---------------------------------------------------------------- */
