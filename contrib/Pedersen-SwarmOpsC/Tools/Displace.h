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
 *	Used for displacing the global optimum of an optimization
 *	problem. This is useful if it is suspected the optimization
 *	method has a strong correlation towards finding optima with
 *	certain values, e.g. zero.
 *
 * ================================================================ */

#ifndef SO_DISPLACE_H
#define SO_DISPLACE_H

#include <SwarmOps/Tools/Types.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Return (x[i]-kDisplaceValue) if kDisplaceOptimum is non-zero,
	 * otherwise just return x[i]. */
	SO_TElm SO_Displace(const SO_TElm *x, SO_TDim i, const int kDisplaceOptimum, const SO_TElm kDisplaceValue);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_DISPLACE_H */
