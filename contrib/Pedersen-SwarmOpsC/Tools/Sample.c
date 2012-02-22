/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Sample
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Tools/Sample.h>
#include <SwarmOps/Tools/Random.h>
#include <SwarmOps/Tools/Macros.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

SO_TElm SO_SampleBoundedOne(const SO_TElm x, const SO_TElm range, const SO_TElm lowerBound, const SO_TElm upperBound)
{
	SO_TElm l, u;	/* Sampling range. */
	SO_TElm y;		/* Actual sample. */

	assert(lowerBound < upperBound);
	assert(range>=0);

	/* Adjust sampling range so it does not exceed bounds. */
	l = SO_Max(x-range, lowerBound);
	u = SO_Min(x+range, upperBound);

	/* Pick a sample from within the bounded range.
	 * Since sampling range is properly bounded,
	 * there is no need to call SO_Bound() afterwards. */
	y = SO_RandBetween(l, u);

	return y;
}

/* ---------------------------------------------------------------- */
