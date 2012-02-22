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
 *	Functions for constraining a vector to the search-space
 *	boundaries.
 *
 * ================================================================ */

#ifndef SO_BOUND_H
#define SO_BOUND_H

#include <SwarmOps/Tools/Types.h>

#ifdef  __cplusplus
extern "C" {
#endif

	/* ---------------------------------------------------------------- */

	/* Return a bounded scalar value. If x is within the lower- or upper-
	 * boundary values, then x itself is returned. Otherwise the appropriate
	 * boundary value is returned. */
	SO_TElm SO_Bound(const SO_TElm x, const SO_TElm lower, const SO_TElm upper);

	/* Bound a single element of a vector. Similar to SO_Bound() for that element. */
	void SO_BoundOne(SO_TElm *x, SO_TDim i, const SO_TElm *lower, const SO_TElm *upper);

	/* Bound all elements of a vector. Similar to SO_Bound() for each element. */
	void SO_BoundAll(SO_TElm *x, SO_TDim n, const SO_TElm *lower, const SO_TElm *upper);

	/*----------------------------------------------------------------*/

#ifdef  __cplusplus
} /* extern "C" end */
#endif

#endif /* #ifndef SO_BOUND_H */
