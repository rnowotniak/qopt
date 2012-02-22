/* ================================================================
 *
 *	SwarmOps - Black-Box Optimization in ANSI C.
 *	Copyright (C) 2003-2008 Magnus Erik Hvass Pedersen.
 *	Published under the GNU Lesser General Public License.
 *	Please see the file license.txt for license details.
 *	SwarmOps on the internet: http://www.Hvass-Labs.org/
 *
 *	Displace
 *
 *	Please see header-file for description.
 *
 * ================================================================ */

#include <SwarmOps/Tools/Displace.h>
#include <math.h>
#include <assert.h>

/* ---------------------------------------------------------------- */

SO_TElm SO_Displace(const SO_TElm *x, SO_TDim i, const int kDisplaceOptimum, const SO_TElm kDisplaceValue)
{
	return (kDisplaceOptimum) ? (x[i]-kDisplaceValue) : (x[i]);
}

/* ---------------------------------------------------------------- */
